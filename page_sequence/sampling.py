"""Resumable training-only coverage and anchored sampling, independent of vector/model code."""
from collections import Counter, defaultdict, OrderedDict
import copy
import math

import numpy as np

from .schema import Window, join_stratum
from .cardinality import PairCardinality

MIXTURES = {'pretrain': (0.60, 0.25, 0.15), 'finetune': (0.20, 0.15, 0.65)}
SOURCES = ('coverage', 'interesting', 'annotated')


class WindowSampler:
    def __init__(self, index, window_size=128, seed=17, stage='pretrain', variable_lengths=False, edge_crops=None):
        if window_size < 2 or stage not in MIXTURES:
            raise ValueError('window_size must be >=2 and stage pretrain/finetune')
        self.index, self.window_size, self.stage = index, window_size, stage
        self.variable_lengths = bool(variable_lengths)
        self.edge_crops = dict(edge_crops or {})
        if (set(self.edge_crops) - {'start_probability', 'end_probability'}
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.edge_crops.values())
                or sum(self.edge_crops.values()) > 1):
            raise ValueError('edge crop probabilities must be finite, nonnegative and sum to at most 1')
        self.draw_size = window_size
        self.docs = index.documents('fit')
        if not self.docs:
            raise ValueError('no fit documents')
        self.by_id = {d.id: d for d in self.docs}
        self.volumes, self.periodicals = defaultdict(list), defaultdict(list)
        for d in self.docs:
            if d.volume:
                self.volumes[d.library, d.volume].append(d)
            if d.periodical:
                self.periodicals[d.library, d.periodical].append(d)
        self.hard_volumes = [v for v in self.volumes.values() if len(v) > 1]
        self.anchor_counts = {s: dict(index.db.execute(
            'SELECT label,COUNT(*) FROM anchors WHERE source=? GROUP BY label', (s,)))
            for s in ('interesting', 'annotated')}
        # Derived, bounded caches: not sampler state and never consume RNG draws.
        self._page_sets = OrderedDict()
        self._cached_pages = 0
        self._cross_volume_pools = OrderedDict()
        self._cached_neighbors = 0
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.coverage_order = []
        self.coverage_cursor = self.coverage_offset = 0
        self.counters = Counter()
        self.seen_volumes = set()  # bounded by the catalogue's volume count
        self.pair_cardinality = PairCardinality()
        self.pending_sources = []
        self._new_coverage_epoch()

    def _new_coverage_epoch(self):
        # Shuffle volume runs, and sibling order inside each run. Every document appears once.
        groups = defaultdict(list)
        for d in self.docs:
            groups[(d.library, 'volume', d.volume) if d.volume else ('doc', d.id)].append(d.id)
        runs = list(groups.values())
        self.rng.shuffle(runs)
        self.coverage_order = []
        for run in runs:
            self.rng.shuffle(run)
            self.coverage_order.extend(run)
        self.coverage_cursor = self.coverage_offset = 0

    def _compatible(self, a, b):
        if a.id == b.id or a.copy_group == b.copy_group:
            return False
        # Fetch contiguous document metadata once instead of probing the corpus
        # UUID index for every page of every candidate pair. Exact set intersection
        # preserves the original shared-page exclusion across libraries too.
        if max(a.length, b.length) <= 65536:
            return self._page_set(a).isdisjoint(self._page_set(b))
        # Keep the rare oversized-document path bounded; scan the shorter side.
        if a.length > b.length:
            a, b = b, a
        return not self.index.db.execute(
            'SELECT 1 FROM pages a JOIN pages b ON a.page=b.page '
            'WHERE a.document=? AND b.document=? LIMIT 1', (a.id, b.id)).fetchone()

    def _page_set(self, doc):
        if doc.id in self._page_sets:
            self._page_sets.move_to_end(doc.id)
            return self._page_sets[doc.id]
        pages = frozenset(r[0] for r in self.index.db.execute(
            'SELECT page FROM pages WHERE document=?', (doc.id,)))
        while self._page_sets and (self._cached_pages + len(pages) > 65536
                                   or len(self._page_sets) >= 128):
            _, removed = self._page_sets.popitem(last=False)
            self._cached_pages -= len(removed)
        self._page_sets[doc.id] = pages
        self._cached_pages += len(pages)
        return pages

    def _cross_volume_pool(self, doc):
        key = doc.library, doc.periodical, doc.volume
        if key in self._cross_volume_pools:
            self._cross_volume_pools.move_to_end(key)
            return self._cross_volume_pools[key]
        pool = [d for d in self.periodicals.get((doc.library, doc.periodical), [])
                if join_stratum(doc, d) == 'cross_volume'] if doc.periodical else []
        if len(pool) <= 65536:
            while self._cross_volume_pools and (self._cached_neighbors + len(pool) > 65536
                                               or len(self._cross_volume_pools) >= 128):
                _, removed = self._cross_volume_pools.popitem(last=False)
                self._cached_neighbors -= len(removed)
            self._cross_volume_pools[key] = pool
            self._cached_neighbors += len(pool)
        return pool

    def _neighbor(self, doc, excluded):
        requested = self.rng.choice(('same_volume', 'cross_volume', 'unrestricted'), p=(.6, .2, .2))
        # Explicit fallback accounting. Random probing avoids scanning every document each draw.
        for stratum in (requested, 'unrestricted') if requested != 'unrestricted' else (requested,):
            # Build only the requested pool, in exactly the original document order.
            # In particular, most draws do not need a cross-volume list at all.
            if stratum == 'same_volume':
                pool = self.volumes.get((doc.library, doc.volume), []) if doc.volume else []
            elif stratum == 'cross_volume':
                pool = self._cross_volume_pool(doc)
            else:
                pool = self.docs
            if not pool:
                continue
            for _ in range(min(len(pool) * 2, 64)):
                d = pool[int(self.rng.integers(len(pool)))]
                if d.id not in excluded and self._compatible(doc, d):
                    if stratum != requested:
                        self.counters['join_fallback:' + requested] += 1
                    self.counters['requested_join:' + requested] += 1
                    return d
        self.counters['no_eligible_neighbor'] += 1
        return None

    def _coverage(self):
        rows = []
        previous = None
        while len(rows) < self.draw_size:
            if self.coverage_cursor == len(self.coverage_order):
                # Do not splice the end of one coverage pass to its beginning.
                self.epoch += 1
                self._new_coverage_epoch()
                if rows:
                    break
                continue
            d = self.by_id[self.coverage_order[self.coverage_cursor]]
            if previous is not None and not self._compatible(previous, d):
                break
            n = min(d.length - self.coverage_offset, self.draw_size - len(rows))
            rows.extend(range(d.first + self.coverage_offset, d.first + self.coverage_offset + n))
            self.coverage_offset += n
            if self.coverage_offset == d.length:
                self.coverage_cursor += 1
                self.coverage_offset = 0
                previous = d
            else:
                break
        self.counters['coverage_pages'] += len(rows)
        return rows

    def _anchor(self, source):
        counts = self.anchor_counts[source]
        if not counts:
            self.counters['source_fallback:' + source] += 1
            return self._coverage()
        labels = list(counts)
        total = sum(counts.values())
        masses = np.array([counts[c] * min(np.sqrt(total / counts[c]), 10) for c in labels])
        label = int(self.rng.choice(labels, p=masses / masses.sum()))
        slot = int(self.rng.integers(counts[label]))
        row, doc_id = self.index.db.execute(
            'SELECT a.row_id,p.document FROM anchors a JOIN pages p ON p.id=a.row_id '
            'WHERE a.source=? AND a.label=? AND a.slot=?', (source, label, slot)).fetchone()
        return self._around(self.by_id[doc_id], row)

    def _around(self, doc, anchor, boundary=False):
        # A one-page scan cannot supervise an adjacent-page boundary.
        left = (0 if self.draw_size == 1 else
                int(self.rng.integers(1 if boundary else 0, self.draw_size)))
        segments = [doc]
        anchor_pos = anchor - doc.first
        used = {doc.id}
        while anchor_pos < left:
            d = self._neighbor(segments[0], used)
            if d is None:
                break
            segments.insert(0, d)
            used.add(d.id)
            anchor_pos += d.length
        start = max(0, anchor_pos - left)
        length = sum(d.length for d in segments)
        while length < start + self.draw_size:
            d = self._neighbor(segments[-1], used)
            if d is None:
                break
            segments.append(d)
            used.add(d.id)
            length += d.length
        rows = []
        position = 0
        end = min(length, start + self.draw_size)
        for d in segments:
            lo, hi = max(0, start - position), min(d.length, end - position)
            if hi > lo:
                rows.extend(range(d.first + lo, d.first + hi))
            position += d.length
        return rows

    def _boundary(self):
        # Uniform over volumes first; large convenient volumes do not dominate hard anchors.
        if self.hard_volumes and self.rng.random() < .6:
            pool = self.hard_volumes[int(self.rng.integers(len(self.hard_volumes)))]
        else:
            pool = self.docs
        d = pool[int(self.rng.integers(len(pool)))]
        return self._around(d, d.first, boundary=True)

    def _draw_length(self):
        if not self.variable_lengths:
            return self.window_size
        # Fixed-length majority; ranges adapt to the configured maximum.
        ranges = ((self.window_size, self.window_size, .60),
                  (32, self.window_size - 1, .20), (8, 31, .10), (2, 7, .05), (1, 1, .05))
        available = [(lo, min(hi, self.window_size), p) for lo, hi, p in ranges
                     if lo <= min(hi, self.window_size)]
        probabilities = np.array([p for _, _, p in available])
        lo, hi, _ = available[int(self.rng.choice(len(available), p=probabilities / probabilities.sum()))]
        return int(self.rng.integers(lo, hi + 1)) if lo != hi else lo

    def draw(self):
        self.draw_size = self._draw_length()
        self.counters[f'requested_length:{self.draw_size}'] += 1
        if not self.pending_sources:
            # Exact source quotas per 100 draws, shuffled; state survives stage transitions/resume.
            self.pending_sources = [s for s, p in zip(SOURCES, MIXTURES[self.stage]) for _ in range(round(p * 100))]
            self.rng.shuffle(self.pending_sources)
        source = self.pending_sources.pop()
        rows = (self._coverage() if source == 'coverage' else self._anchor('annotated')
                if source == 'annotated' else self._anchor('interesting')
                if self.rng.random() < 2 / 3 else self._boundary())
        pages, document_ids = self.index.pages_with_documents(rows)
        if source != 'coverage' and any(self.edge_crops.values()):
            draw = self.rng.random()
            start_p = self.edge_crops.get('start_probability', 0)
            kind = ('start' if draw < start_p else 'end'
                    if draw < start_p + self.edge_crops.get('end_probability', 0) else None)
            if kind:
                self.counters['edge_crop_requested:' + kind] += 1
                gold = [i for i, p in enumerate(pages) if p.annotated_type]
                candidates = [i for i, (row, docid) in enumerate(zip(rows, document_ids))
                              if row == (self.by_id[docid].first if kind == 'start' else
                                         self.by_id[docid].first + self.by_id[docid].length - 1)
                              and (not gold or (i <= min(gold) if kind == 'start' else i >= max(gold)))]
                if candidates:
                    i = int(self.rng.choice(candidates))
                    section = slice(i, None) if kind == 'start' else slice(None, i + 1)
                    rows, pages, document_ids = rows[section], pages[section], document_ids[section]
                    self.counters['edge_crop_applied:' + kind] += 1
                else:
                    self.counters['edge_crop_unavailable:' + kind] += 1
        self.counters['windows_starting_at_document_start'] += int(rows[0] == self.by_id[document_ids[0]].first)
        last_doc = self.by_id[document_ids[-1]]
        self.counters['windows_ending_at_document_end'] += int(rows[-1] == last_doc.first + last_doc.length - 1)
        self.counters['source:' + source] += 1
        self.counters['processed_pages'] += len(rows)
        self.counters[f'actual_length:{len(rows)}'] += 1
        self.counters['windows_with_gold'] += int(any(p.annotated_type for p in pages))
        for p in pages:
            label, gold, _ = p.target()
            if label >= 0:
                self.counters[f'type_exposure:{"gold" if gold else "weak"}:{label}'] += 1
        self.counters['windows_with_rare_type'] += int(any(
            (p.annotated_type or p.db_type) not in ('', 'NormalPage') for p in pages))
        starts = 0
        for da, db in zip(document_ids, document_ids[1:]):
            if da != db:
                starts += 1
                self.counters['visible_join:' + join_stratum(self.by_id[da], self.by_id[db])] += 1
                self.pair_cardinality.add((da, db))
                for d in (self.by_id[da], self.by_id[db]):
                    if d.volume:
                        self.seen_volumes.add((d.library, d.volume))
        self.counters['unique_join_pairs_estimate'] = self.pair_cardinality.estimate()
        self.counters['unique_join_volumes'] = len(self.seen_volumes)
        self.counters['windows_boundaries:' + ('multiple' if starts > 1 else str(starts))] += 1
        return Window(tuple(rows), source)

    def set_stage(self, stage):
        if stage not in MIXTURES:
            raise ValueError(stage)
        self.stage = stage
        self.pending_sources = []

    def state_dict(self):
        # Copy flat containers only; deep-copying millions of pair tuples caused
        # multi-GiB checkpoint spikes. Cardinality snapshots are fixed-size bytes.
        return {**{k: copy.copy(getattr(self, k)) for k in (
            'window_size', 'variable_lengths', 'stage', 'epoch', 'coverage_order', 'coverage_cursor',
            'coverage_offset', 'counters', 'pending_sources', 'seen_volumes')},
            'pair_cardinality': self.pair_cardinality.state_dict(),
            'rng': copy.deepcopy(self.rng.bit_generator.state), 'edge_crops': self.edge_crops.copy()}

    def load_state_dict(self, state):
        if state.get('variable_lengths', False) != self.variable_lengths:
            raise ValueError('sampler length policy mismatch')
        if state['window_size'] != self.window_size:
            raise ValueError('sampler window size mismatch')
        if state.get('edge_crops', {}) != self.edge_crops:
            raise ValueError('sampler edge crop policy mismatch')
        for k, v in state.items():
            if k == 'rng':
                self.rng.bit_generator.state = copy.deepcopy(v)
            elif k in ('seen_pairs', 'pair_cardinality'):
                continue
            else:
                setattr(self, k, Counter(v) if k == 'counters' else copy.copy(v))
        if 'pair_cardinality' in state:
            self.pair_cardinality = PairCardinality(state['pair_cardinality'])
        else:
            self.pair_cardinality = PairCardinality()
            for pair in state.get('seen_pairs', ()):
                self.pair_cardinality.add(pair)
        self.counters.pop('unique_join_pairs', None)
        self.counters['unique_join_pairs_estimate'] = self.pair_cardinality.estimate()
