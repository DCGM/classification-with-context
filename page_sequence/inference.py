"""Arbitrary-length inference and page/pair-aware merging; no backbone imports."""
from dataclasses import dataclass

import numpy as np
import torch

from .data import PageListDataset, collate_windows, to_device
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

    def __post_init__(self):
        if not (self.window_size >= 2 and 1 <= self.stride < self.window_size):
            raise ValueError('stride must be 1..window_size-1 to cover every adjacent pair')
        if self.merge not in ('center_logits', 'uniform_logits', 'uniform_probs', 'center_probs', 'central'):
            raise ValueError('unknown merge strategy')
        if self.edge_ramp < 1 or self.batch_size < 1 or not 0 < self.start_threshold < 1:
            raise ValueError('invalid inference settings')
        if self.tail not in ('anchored', 'regular'):
            raise ValueError('tail must be anchored or regular')


def window_spans(n, size, stride, tail='anchored'):
    if n < 1 or size < 2 or not 1 <= stride < size:
        raise ValueError('nonempty stream and pair-covering window/stride required')
    if tail not in ('anchored', 'regular'):
        raise ValueError('tail must be anchored or regular')
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(starts[-1] + stride if tail == 'regular' else n - size)
    return [(s, min(n, s + size)) for s in starts]


def context_weights(start, end, n, ramp):
    r = np.arange(end - start)
    # True stream endpoints have no missing outside context to penalize.
    left = r + 1 if start > 0 else np.full(end - start, np.inf)
    right = end - start - r if end < n else np.full(end - start, np.inf)
    return (.1 + .9 * np.minimum(1, np.minimum(left, right) / ramp)).astype(np.float32)


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
        w = context_weights(start, end, self.n, c.edge_ramp)
        if c.merge.startswith('uniform'):
            w[:] = 1
        pair_w = np.r_[0, np.minimum(w[:-1], w[1:])].astype(np.float32)
        p = np.asarray(output['pair_mask'], bool)
        if p.shape != (end - start,) or p[0] or not p[1:].all():
            raise ValueError('invalid pair coverage in an unpadded inference window')
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
            self.window_count[start:end] += 1
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
        if not (self.page_weight > 0).all() or not (self.pair_weight[1:] > 0).all():
            raise ValueError('missing page or adjacent-pair predictions')
        scores = {}
        for task, values in self.sums.items():
            weights = self.pair_weight if task == 'starts' else self.page_weight
            scores[task] = (values.copy() if self.config.merge == 'central' else
                            values / np.maximum(weights if values.ndim == 1 else weights[:, None], 1e-12))
        probs = scores if self.config.merge.endswith('probs') else {
            task: sigmoid(x) if x.ndim == 1 else softmax(x) for task, x in scores.items()}
        probs['starts'][0] = np.nan
        pair_mask = np.ones(self.n, bool)
        pair_mask[0] = False
        starts = pair_mask & (probs['starts'] >= self.config.start_threshold)
        result = {'type_probs': probs['type'], 'placement_probs': probs['placement'],
                'starts_document_probs': probs['starts'], 'pair_mask': pair_mask,
                'type': probs['type'].argmax(-1), 'placement': probs['placement'].argmax(-1),
                'starts': starts, 'segment_ids': np.cumsum(starts),
                'page_weight': self.page_weight, 'pair_weight': self.pair_weight}
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
    borders = getattr(model, 'scan_borders', False)
    boundary_features = getattr(model, 'boundary_features', False)
    if borders or boundary_features:
        from .scan_borders import scan_windows
        tiles = scan_windows(len(rows), config.window_size, config.stride,
                             border_tokens=borders, tail=config.tail)
    else:
        tiles = [(s, e, False, False) for s, e in
                 window_spans(len(rows), config.window_size, config.stride, config.tail)]
    supplied = None
    if document_starts is not None or document_ends is not None:
        if not boundary_features:
            raise ValueError('document metadata requires five-bit boundary features')
        from .scan_borders import encode_boundary_features
        supplied = encode_boundary_features(len(rows), document_starts=document_starts,
                                            document_ends=document_ends)
    spans = [(s, e) for s, e, _, _ in tiles]
    merger = StreamMerger(len(rows), model.num_types, config,
                          collect_embeddings, collect_window_stats)
    was_training = model.training
    model.eval()
    try:
        for first in range(0, len(spans), config.batch_size):
            current = spans[first:first + config.batch_size]
            examples = []
            for s, e, begin, end in tiles[first:first + config.batch_size]:
                example = dataset[Window(tuple(rows[s:e]), scan_start=begin, scan_end=end)]
                if borders and 'scan_borders' not in example['features']:
                    flags = np.zeros((e - s, 2), np.float32)
                    flags[0, 0], flags[-1, 1] = begin, end
                    example['features'] = dict(example['features'], scan_borders=flags)
                if boundary_features and 'boundaries' not in example['features']:
                    from .scan_borders import encode_boundary_features
                    values = encode_boundary_features(e - s, begin, end)
                    example['features'] = dict(example['features'], boundaries=values)
                if supplied is not None:
                    values = supplied[s:e].copy()
                    values[0, 0], values[-1, 1] = begin, end
                    example['features'] = dict(example['features'], boundaries=values)
                examples.append(example)
            batch = to_device(collate_windows(examples), device,
                              feature_dtype=torch.float32 if getattr(model, 'compute_precision', 'float32')
                              == 'float32' else None)
            with autocast(model, device):
                kwargs = {'return_embeddings': True} if collect_embeddings else {}
                output = model(batch['features'], batch['attention_mask'], **kwargs)
            for i, (s, e) in enumerate(current):
                merger.add(s, e, {k: v[i, :e - s].detach().float().cpu().numpy() for k, v in output.items()})
    finally:
        model.train(was_training)
    result = merger.finalize()
    result['processed_pages'] = sum(e - s for s, e in spans)
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
