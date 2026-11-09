"""Arbitrary-length inference and page/pair-aware merging; no backbone imports."""
from dataclasses import dataclass

import numpy as np
import torch

from .data import PageListDataset, to_device
from .schema import Window
from .precision import autocast


@dataclass(frozen=True)
class InferenceConfig:
    window_size: int = 128
    stride: int = 64
    merge: str = 'center_logits'
    edge_ramp: int = 32
    start_threshold: float = .5
    batch_size: int = 8
    tail: str = 'regular'
    edge_ignore: int = 0
    edge_power: int = 2

    def __post_init__(self):
        if not ((self.window_size == 1 and self.stride == 1) or
                (self.window_size >= 2 and 1 <= self.stride < self.window_size)):
            raise ValueError('stride must be 1..window_size-1 to cover every adjacent pair')
        if self.merge not in ('center_logits', 'uniform_logits', 'uniform_probs', 'center_probs', 'central'):
            raise ValueError('unknown merge strategy')
        if self.edge_ramp < 1 or self.batch_size < 1 or not 0 < self.start_threshold < 1:
            raise ValueError('invalid inference settings')
        if not 0 <= 2 * self.edge_ignore <= self.window_size - self.stride:
            raise ValueError('edge_ignore must be nonnegative and at most half the overlap')
        if type(self.edge_power) is not int or self.edge_power < 1:
            raise ValueError('edge_power must be a positive integer')
        if self.edge_ignore >= self.edge_ramp:
            raise ValueError('edge_ignore must be smaller than edge_ramp')
        if self.tail not in ('anchored', 'regular'):
            raise ValueError('tail must be anchored or regular')


def deployment_inference(window_size, batch_size=8, start_threshold=.5):
    """The merging policy deployed in MetaKat, scaled to a window: overlap W/8 (64 at W512, at least one page),
    regular tail, centre-weighted cubic logits whose ramp ends at 3/4 of the overlap with 3/16 of the overlap
    ignored at artificial window edges, rounded like MetaKat's `overlap_merge`. W1 has nothing to merge."""
    if window_size == 1:
        return InferenceConfig(window_size=1, stride=1, merge='uniform_logits', edge_ramp=1, edge_ignore=0,
                               edge_power=1, tail='regular', start_threshold=start_threshold, batch_size=batch_size)
    overlap = max(1, window_size // 8)
    return InferenceConfig(window_size=window_size, stride=window_size - overlap, merge='center_logits',
                           edge_ramp=max(1, int(overlap * .75)), edge_ignore=int(overlap * .1875), edge_power=3,
                           tail='regular', start_threshold=start_threshold, batch_size=batch_size)


def window_spans(n, size, stride, tail='anchored'):
    if size == 1 and stride == 1 and n >= 1:
        return [(i, i + 1) for i in range(n)]
    # stride == size tiles without overlap (training scans); inference configurations require stride < size.
    if n < 1 or size < 2 or not 1 <= stride <= size:
        raise ValueError('nonempty stream and a stride of at most the window required')
    if tail not in ('anchored', 'regular'):
        raise ValueError('tail must be anchored or regular')
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(starts[-1] + stride if tail == 'regular' else n - size)
    return [(s, min(n, s + size)) for s in starts]


def context_weights(start, end, n, ramp, ignore=0, power=2):
    r = np.arange(end - start)
    # True stream endpoints have no missing outside context to penalize.
    left = r + 1 if start > 0 else np.full(end - start, np.inf)
    right = end - start - r if end < n else np.full(end - start, np.inf)
    distance = np.minimum(left, right)
    if ignore:
        # Exactly `ignore` pages at artificial edges contribute nothing. Real
        # scan endpoints are exempt, including the short final window. The
        # ramp still reaches full weight at the original distance `ramp`.
        if ignore >= ramp:
            raise ValueError('ignore must be smaller than ramp')
        return np.power(np.clip((distance - ignore) / (ramp - ignore), 0, 1), power).astype(np.float32)
    return np.power(np.minimum(1, distance / ramp), power).astype(np.float32)


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class StreamMerger:
    """Float32 accumulators; optional representations/stats, no retained windows."""
    def __init__(self, n, num_types, config, collect_embeddings=False, collect_window_stats=False):
        self.n, self.config = n, config
        self.collect_embeddings, self.collect_window_stats = collect_embeddings, collect_window_stats
        self.embedding_sum = None
        self.optional_outputs = None
        self.sums = {'type': np.zeros((n, num_types), np.float32),
                     'placement': np.zeros((n, 3), np.float32), 'starts': np.zeros(n, np.float32)}
        self.page_weight = np.zeros(n, np.float32)
        self.pair_weight = np.zeros(n, np.float32)
        if collect_window_stats:
            self.window_count = np.zeros(n, np.int32)
            self.stats_weight = np.zeros(n, np.float32)
            self.prob_sum = np.zeros((n, num_types), np.float32)
            self.entropy_sum = np.zeros(n, np.float32)

    def add(self, start, end, output):
        if not 0 <= start < end <= self.n:
            raise ValueError('window outside stream')
        c = self.config
        w = context_weights(start, end, self.n, c.edge_ramp, c.edge_ignore, c.edge_power)
        if c.merge.startswith('uniform'):
            w[w > 0] = 1
        pair_w = np.r_[0, np.minimum(w[:-1], w[1:])].astype(np.float32)
        if c.edge_ignore:
            # When trimming half the overlap, retained regions meet between
            # two pages. Keep that seam pair if either endpoint is retained;
            # pairs with both endpoints inside the discarded region stay zero.
            seam = (w[:-1] == 0) != (w[1:] == 0)
            pair_w[1:][seam] = np.maximum(w[:-1], w[1:])[seam]
        p = np.asarray(output['pair_mask'], bool)
        singleton = c.window_size == 1
        if p.shape != (end - start,) or (p.any() if singleton else p[0] or not p[1:].all()):
            raise ValueError('invalid pair coverage in an unpadded inference window')
        if singleton:
            pair_w[:] = 0
        optional = tuple(k for k in ('page_start', 'page_end') if k + '_logits' in output)
        if self.optional_outputs is None:
            self.optional_outputs = optional
            for task in optional:
                self.sums[task] = np.zeros(self.n, np.float32)
        elif optional != self.optional_outputs:
            raise ValueError('auxiliary outputs changed between windows')
        tasks = [('type', 'type_logits'), ('placement', 'placement_logits'),
                 ('starts', 'starts_document_logits')] + [(k, k + '_logits') for k in optional]
        for task, key in tasks:
            x = np.asarray(output[key], np.float32)
            if x.shape != self.sums[task][start:end].shape or not np.isfinite(x).all():
                raise ValueError('invalid window outputs')
            if c.merge.endswith('probs'):
                x = sigmoid(x) if x.ndim == 1 else softmax(x)
            weight = pair_w if task == 'starts' else w
            if c.merge == 'central':
                previous = self.pair_weight[start:end] if task == 'starts' else self.page_weight[start:end]
                take = weight > previous  # deterministic earlier-window tie break
                self.sums[task][start:end][take] = x[take]
            else:
                self.sums[task][start:end] += x * (weight if x.ndim == 1 else weight[:, None])
        if self.collect_embeddings:
            x = np.asarray(output['embeddings'], np.float32)
            if (x.ndim != 2 or x.shape[0] != end - start or x.shape[1] < 1
                    or not np.isfinite(x).all()):
                raise ValueError('invalid window embeddings')
            if self.embedding_sum is None:
                self.embedding_sum = np.zeros((self.n, x.shape[1]), np.float32)
            if x.shape[1] != self.embedding_sum.shape[1]:
                raise ValueError('embedding width changed between windows')
            if c.merge == 'central':
                take = w > self.page_weight[start:end]
                self.embedding_sum[start:end][take] = x[take]
            else:
                self.embedding_sum[start:end] += x * w[:, None]
        if self.collect_window_stats:
            probabilities = softmax(np.asarray(output['type_logits'], np.float32))
            entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(-1)
            self.window_count[start:end] += (w > 0)
            self.stats_weight[start:end] += w
            self.prob_sum[start:end] += probabilities * w[:, None]
            self.entropy_sum[start:end] += entropy * w
        if c.merge == 'central':
            self.page_weight[start:end] = np.maximum(self.page_weight[start:end], w)
            self.pair_weight[start:end] = np.maximum(self.pair_weight[start:end], pair_w)
        else:
            self.page_weight[start:end] += w
            self.pair_weight[start:end] += pair_w

    def finalize(self):
        enabled = self.config.window_size != 1
        if not (self.page_weight > 0).all() or (enabled and not (self.pair_weight[1:] > 0).all()):
            raise ValueError('missing page or adjacent-pair predictions')
        scores = {}
        for task, values in self.sums.items():
            weights = self.pair_weight if task == 'starts' else self.page_weight
            scores[task] = (values.copy() if self.config.merge == 'central' else
                            values / np.maximum(weights if values.ndim == 1 else weights[:, None], 1e-12))
        probs = scores if self.config.merge.endswith('probs') else {
            task: sigmoid(x) if x.ndim == 1 else softmax(x) for task, x in scores.items()}
        probs['starts'][0 if enabled else slice(None)] = np.nan
        pair_mask = np.full(self.n, enabled, bool)
        pair_mask[0] = False
        starts = pair_mask & (probs['starts'] >= self.config.start_threshold)
        result = {'type_probs': probs['type'], 'placement_probs': probs['placement'],
                'starts_document_probs': probs['starts'], 'pair_mask': pair_mask,
                'type': probs['type'].argmax(-1), 'placement': probs['placement'].argmax(-1),
                'starts': starts, 'segment_ids': np.cumsum(starts),
                'page_weight': self.page_weight, 'pair_weight': self.pair_weight}
        if not enabled:
            result['boundary_enabled'] = False
        for task in self.optional_outputs or ():
            result[task + '_probs'] = probs[task]
        if self.collect_embeddings:
            result['embeddings'] = (self.embedding_sum.copy() if self.config.merge == 'central' else
                                    self.embedding_sum / self.page_weight[:, None])
        if self.collect_window_stats:
            mean = self.prob_sum / self.stats_weight[:, None]
            entropy = -(mean * np.log(np.maximum(mean, 1e-30))).sum(-1)
            disagreement = np.maximum(0, entropy - self.entropy_sum / self.stats_weight)
            disagreement[self.window_count == 1] = 0
            result.update(window_count=self.window_count, type_window_js=disagreement)
        return result


@torch.inference_mode()
def predict_stream(model, dataset, rows, config=InferenceConfig(), device='cpu',
                   document_starts=None, document_ends=None,
                   collect_embeddings=False, collect_window_stats=False):
    """Tile the stream into windows and predict them in batches of up to `config.batch_size` windows.

    One code path for every window size, W1 included. As in training, each batch reads its contiguous stretch
    of the stream once and its windows are cut from it in memory; only the scan flags differ per window. The
    outputs are copied to the host once per batch and merged window by window.
    """
    n = len(rows)
    borders = getattr(model, 'scan_borders', False)
    boundary_features = getattr(model, 'boundary_features', False)
    if borders or boundary_features:
        from .scan_borders import scan_windows
        tiles = scan_windows(n, config.window_size, config.stride, border_tokens=borders, tail=config.tail)
    else:
        tiles = [(s, e, False, False) for s, e in window_spans(n, config.window_size, config.stride, config.tail)]
    supplied = None
    if document_starts is not None or document_ends is not None:
        if not boundary_features:
            raise ValueError('document metadata requires five-bit boundary features')
        from .scan_borders import encode_boundary_features
        supplied = torch.from_numpy(encode_boundary_features(n, document_starts=document_starts,
                                                             document_ends=document_ends))
    merger = StreamMerger(n, model.num_types, config, collect_embeddings, collect_window_stats)
    feature_dtype = torch.float32 if getattr(model, 'compute_precision', 'float32') == 'float32' else None
    was_training = model.training
    model.eval()
    try:
        for first in range(0, len(tiles), config.batch_size):
            current = tiles[first:first + config.batch_size]
            lo, hi = min(t[0] for t in current), max(t[1] for t in current)
            block = dataset[Window(tuple(rows[lo:hi]), scan_start=lo == 0, scan_end=hi == n)]['features']
            sizes = torch.tensor([e - s for s, e, _, _ in current])
            positions = torch.arange(int(sizes.max()))
            mask = positions[None] < sizes[:, None]
            index = (torch.tensor([s - lo for s, _, _, _ in current])[:, None] + positions[None]).clamp_max(hi - lo - 1)
            features = {name: torch.from_numpy(np.ascontiguousarray(x))[index].masked_fill(~mask[..., None], 0)
                        for name, x in block.items() if name not in ('scan_borders', 'boundaries')}
            # Scan flags belong to each window: its first page may open the scan, its last may close it.
            begin = torch.tensor([float(b) for _, _, b, _ in current])
            end = torch.tensor([float(e) for _, _, _, e in current])
            last = (torch.arange(len(current)), sizes - 1)
            if borders or 'scan_borders' in block:
                flags = torch.zeros(len(current), len(positions), 2)
                flags[:, 0, 0], flags[last + (1,)] = begin, end
                features['scan_borders'] = flags
            if boundary_features or 'boundaries' in block:
                values = (supplied[lo:hi][index].masked_fill(~mask[..., None], 0) if supplied is not None
                          else torch.zeros(len(current), len(positions), 5))
                values[:, 0, 0], values[last + (1,)] = begin, end
                features['boundaries'] = values
            batch = to_device({'features': features, 'attention_mask': mask}, device, feature_dtype=feature_dtype)
            with autocast(model, device):
                kwargs = {'return_embeddings': True} if collect_embeddings else {}
                output = model(batch['features'], batch['attention_mask'], **kwargs)
            output = {k: v.detach().float().cpu().numpy() for k, v in output.items()}
            for i, (s, e, _, _) in enumerate(current):
                merger.add(s, e, {k: v[i, :e - s] for k, v in output.items()})
    finally:
        model.train(was_training)
    result = merger.finalize()
    result['processed_pages'] = sum(e - s for s, e, _, _ in tiles)
    return result


def tag_pages(model, features, pages, config=InferenceConfig(), device='cpu', pagination_cap=10000,
              document_starts=None, document_ends=None,
              collect_embeddings=False, collect_window_stats=False):
    """Public inference API: caller's scan order, no document index/known boundaries required.

    Pages need library/page vector keys and image paths; doc can be an empty string.
    Repeated UUIDs retain distinct stream positions. First start probability is NaN/invalid.
    """
    if 'pagination' in model.dims:
        from .pagination import PaginationFeatures
        features = PaginationFeatures(features, pagination_cap)
    dataset = PageListDataset(pages, features)
    result = predict_stream(model, dataset, tuple(range(len(pages))), config, device,
                            document_starts=document_starts, document_ends=document_ends,
                            collect_embeddings=collect_embeddings, collect_window_stats=collect_window_stats)
    result['page_keys'] = np.array([p.key for p in pages])
    return result
