"""Sample complete synthetic scan batches, then tile them exactly as inference."""
from collections import deque
import copy

import numpy as np

from .sampling import MIXTURES, SOURCES, WindowSampler
from .scan_borders import scan_windows
from .schema import Window, join_stratum


class ScanBatchSampler(WindowSampler):
    def __init__(self, *args, scan_batches=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.window_size < 3 or any(self.edge_crops.values()):
            raise ValueError('scan borders require >=3 positions and replace legacy edge crops')
        settings = dict(scan_batches or {})
        if set(settings) - {'full_scan_multiplier', 'stride', 'border_tokens', 'metadata_probability', 'tail'}:
            raise ValueError('unknown scan batch setting')
        self.scan_batches = dict(full_scan_multiplier=settings.get('full_scan_multiplier', 4),
                                 stride=settings.get('stride', self.window_size // 2),
                                 border_tokens=settings.get('border_tokens', True))
        if (not isinstance(self.scan_batches['full_scan_multiplier'], int)
                or self.scan_batches['full_scan_multiplier'] < 1
                or not isinstance(self.scan_batches['border_tokens'], bool)
                or not 1 <= self.scan_batches['stride'] <= self.window_size):
            raise ValueError('invalid scan batch multiplier/stride')
        probability = settings.get('metadata_probability', 0.)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('metadata_probability must be finite and in [0, 1]')
        if probability:
            if self.scan_batches['border_tokens']:
                raise ValueError('document metadata requires five-bit features')
            self.scan_batches['metadata_probability'] = probability
        # 'anchored' (the original default) shifts the last window back to end at the scan end; 'regular' keeps
        # the stride and lets the last window be shorter, as at inference. Only the non-default value is stored,
        # so the sampler state of earlier runs stays unchanged.
        tail = settings.get('tail', 'anchored')
        if tail not in ('anchored', 'regular'):
            raise ValueError('scan batch tail must be anchored or regular')
        if tail == 'regular':
            self.scan_batches['tail'] = tail
        self.pending_coverage_windows = deque()

    def _tiles(self, docs, source):
        rows = tuple(row for d in docs for row in range(d.first, d.first + d.length))
        tiles = scan_windows(len(rows), self.window_size, self.scan_batches['stride'],
                             self.scan_batches['border_tokens'], self.scan_batches.get('tail', 'anchored'))
        self.counters['constructed_scans:' + source] += 1
        self.counters['constructed_scan_pages:' + source] += len(rows)
        self.counters['constructed_scan_documents:' + source] += len(docs)
        probability = self.scan_batches.get('metadata_probability', 0.)
        provided = bool(probability and self.rng.random() < probability)
        self.counters['metadata_scans:' + ('provided' if provided else 'absent')] += 1
        return [Window(rows[lo:hi], source, begin, end, provided) for lo, hi, begin, end in tiles]

    def _coverage(self):
        if not self.pending_coverage_windows:
            if self.coverage_cursor == len(self.coverage_order):
                self.epoch += 1
                self._new_coverage_epoch()
            docs, length = [], 0
            while self.coverage_cursor < len(self.coverage_order):
                d = self.by_id[self.coverage_order[self.coverage_cursor]]
                if docs and not self._compatible(docs[-1], d):
                    break
                docs.append(d)
                length += d.length
                self.coverage_cursor += 1
                if length >= self.draw_size:
                    break
            self.counters['coverage_pages'] += length
            self.pending_coverage_windows.extend(self._tiles(docs, 'coverage'))
        return self.pending_coverage_windows.popleft()

    def _around(self, doc, anchor, boundary=False):
        # All selected documents enter the scan in full. Only its inference tiles
        # can crop a document; those tile edges never acquire false scan markers.
        left = int(self.rng.integers(1 if boundary else 0, max(2, self.draw_size)))
        docs, used = [doc], {doc.id}
        anchor_pos = anchor - doc.first
        while anchor_pos < left:
            previous = self._neighbor(docs[0], used)
            if previous is None:
                break
            docs.insert(0, previous)
            used.add(previous.id)
            anchor_pos += previous.length
        length = sum(d.length for d in docs)
        while length < self.draw_size:
            following = self._neighbor(docs[-1], used)
            if following is None:
                break
            docs.append(following)
            used.add(following.id)
            length += following.length
        candidates = [w for w in self._tiles(docs, 'anchored') if anchor in w.rows
                      and (not boundary or w.rows.index(anchor) > 0)]
        # An isolated first page cannot supervise an adjacent-page boundary.
        if not candidates:
            candidates = [w for w in self._tiles(docs, 'boundary_fallback') if anchor in w.rows]
        return candidates[int(self.rng.integers(len(candidates)))]

    def draw(self):
        requested = self._draw_length()
        self.draw_size = (requested * self.scan_batches['full_scan_multiplier']
                          if requested == self.window_size else requested)
        self.counters[f'requested_scan_pages:{self.draw_size}'] += 1
        if not self.pending_sources:
            self.pending_sources = [s for s, p in zip(SOURCES, MIXTURES[self.stage])
                                    for _ in range(round(p * 100))]
            self.rng.shuffle(self.pending_sources)
        source = self.pending_sources.pop()
        window = (self._coverage() if source == 'coverage' else self._anchor('annotated')
                  if source == 'annotated' else self._anchor('interesting')
                  if self.rng.random() < 2 / 3 else self._boundary())
        window = Window(window.rows, source, window.scan_start, window.scan_end, window.document_metadata)
        pages, ids = self.index.pages_with_documents(window.rows)
        first_doc, last_doc = self.by_id[ids[0]], self.by_id[ids[-1]]
        if window.scan_start and window.rows[0] != first_doc.first:
            raise RuntimeError('scan starts inside a document')
        if window.scan_end and window.rows[-1] != last_doc.first + last_doc.length - 1:
            raise RuntimeError('scan ends inside a document')
        tokens = len(window.rows) + (window.scan_start + window.scan_end) * self.scan_batches['border_tokens']
        # Short windows: complete short scans, and with the regular tail also the last tile of a scan.
        short_allowed = window.scan_end and (window.scan_start or self.scan_batches.get('tail') == 'regular')
        if tokens > self.window_size or (tokens < self.window_size and not short_allowed):
            raise RuntimeError('only complete scans or regular scan tails may produce short padded windows')
        self.counters['metadata_windows:' + ('provided' if window.document_metadata else 'absent')] += 1
        self.counters['source:' + source] += 1
        self.counters['processed_pages'] += len(pages)
        self.counters['actual_length:' + str(len(pages))] += 1
        self.counters['network_length:' + str(tokens)] += 1
        self.counters['scan_start_windows'] += window.scan_start
        self.counters['scan_end_windows'] += window.scan_end
        self.counters['complete_scan_windows'] += window.scan_start and window.scan_end
        self.counters['internal_windows'] += not window.scan_start and not window.scan_end
        self.counters['windows_starting_at_document_start'] += window.rows[0] == first_doc.first
        self.counters['windows_ending_at_document_end'] += window.rows[-1] == last_doc.first + last_doc.length - 1
        self.counters['windows_with_gold'] += any(p.annotated_type for p in pages)
        self.counters['windows_with_rare_type'] += any((p.annotated_type or p.db_type) not in ('', 'NormalPage')
                                                    for p in pages)
        for p in pages:
            label, gold, _ = p.target()
            if label >= 0:
                self.counters[f'type_exposure:{"gold" if gold else "weak"}:{label}'] += 1
        starts = 0
        for a, b in zip(ids, ids[1:]):
            if a != b:
                starts += 1
                da, db = self.by_id[a], self.by_id[b]
                self.counters['visible_join:' + join_stratum(da, db)] += 1
                self.pair_cardinality.add((a, b))
                for d in (da, db):
                    if d.volume:
                        self.seen_volumes.add((d.library, d.volume))
        self.counters['unique_join_pairs_estimate'] = self.pair_cardinality.estimate()
        self.counters['unique_join_volumes'] = len(self.seen_volumes)
        self.counters['windows_boundaries:' + ('multiple' if starts > 1 else str(starts))] += 1
        return window

    def state_dict(self):
        return dict(super().state_dict(), scan_batches=self.scan_batches.copy(),
                    pending_coverage_windows=list(self.pending_coverage_windows))

    def load_state_dict(self, state):
        if state.get('scan_batches') != self.scan_batches:
            raise ValueError('scan batch policy mismatch')
        base = {k: v for k, v in state.items() if k not in ('scan_batches', 'pending_coverage_windows')}
        super().load_state_dict(base)
        self.pending_coverage_windows = deque(copy.copy(state['pending_coverage_windows']))
