"""The batched inference path gives exactly the predictions of the earlier per-window reads (and W1 path)."""
import numpy as np
import pytest
import torch

from page_sequence.data import collate_windows, to_device
from page_sequence.inference import InferenceConfig, deployment_inference, predict_stream, sigmoid, softmax, window_spans
from page_sequence.precision import autocast
from page_sequence.scan_borders import ScanBorderDataset
from page_sequence.schema import Window
from page_sequence.tests.test_context_training import model_for, setup_data
from page_sequence.models import BertBackbone, PageTagger


# The implementation before the batched reads, kept verbatim as the reference.
def reference_context_weights(start, end, n, ramp, ignore=0, power=2):
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


class ReferenceMerger:
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
        w = reference_context_weights(start, end, self.n, c.edge_ramp, c.edge_ignore, c.edge_power)
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
def reference_predict_stream(model, dataset, rows, config=InferenceConfig(), device='cpu',
                   document_starts=None, document_ends=None,
                   collect_embeddings=False, collect_window_stats=False):
    if config.window_size == 1:
        return reference_singletons(model, dataset, rows, config, device,
                                  document_starts, document_ends, collect_embeddings, collect_window_stats)
    borders = getattr(model, 'scan_borders', False)
    boundary_features = getattr(model, 'boundary_features', False)
    if borders or boundary_features:
        from page_sequence.scan_borders import scan_windows
        tiles = scan_windows(len(rows), config.window_size, config.stride,
                             border_tokens=borders, tail=config.tail)
    else:
        tiles = [(s, e, False, False) for s, e in
                 window_spans(len(rows), config.window_size, config.stride, config.tail)]
    supplied = None
    if document_starts is not None or document_ends is not None:
        if not boundary_features:
            raise ValueError('document metadata requires five-bit boundary features')
        from page_sequence.scan_borders import encode_boundary_features
        supplied = encode_boundary_features(len(rows), document_starts=document_starts,
                                            document_ends=document_ends)
    spans = [(s, e) for s, e, _, _ in tiles]
    merger = ReferenceMerger(len(rows), model.num_types, config,
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
                    from page_sequence.scan_borders import encode_boundary_features
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


@torch.inference_mode()
def reference_singletons(model, dataset, rows, config, device, document_starts=None,
                       document_ends=None, collect_embeddings=False, collect_window_stats=False):
    """Read contiguous storage blocks, encode each page alone; no invented scan edges."""
    from page_sequence.context_training import split_context
    from page_sequence.scan_borders import encode_boundary_features
    if getattr(model, 'scan_borders', False):
        raise ValueError('singleton inference requires page-aligned boundary features')
    supplied = (encode_boundary_features(len(rows), document_starts=document_starts,
                                        document_ends=document_ends)
                if document_starts is not None or document_ends is not None else None)
    merger = ReferenceMerger(len(rows), model.num_types, config, collect_embeddings, collect_window_stats)
    was_training = model.training
    model.eval()
    try:
        for lo in range(0, len(rows), config.batch_size):
            hi = min(len(rows), lo + config.batch_size)
            example = dataset[Window(tuple(rows[lo:hi]), scan_start=lo == 0, scan_end=hi == len(rows))]
            if getattr(model, 'boundary_features', False):
                values = (supplied[lo:hi].copy() if supplied is not None else
                          encode_boundary_features(hi - lo))
                values[0, 0], values[-1, 1] = lo == 0, hi == len(rows)
                example['features'] = dict(example['features'], boundaries=values)
            batch = to_device(split_context(collate_windows([example]), 1), device,
                              feature_dtype=torch.float32 if getattr(model, 'compute_precision', 'float32')
                              == 'float32' else None)
            with autocast(model, device):
                kwargs = {'return_embeddings': True} if collect_embeddings else {}
                output = model(batch['features'], batch['attention_mask'], **kwargs)
            merger.add(lo, hi, {k: v[:, 0].detach().float().cpu().numpy() for k, v in output.items()})
    finally:
        model.train(was_training)
    result = merger.finalize()
    result['processed_pages'] = len(rows)
    return result


def compare(model, dataset, rows, config, **kwargs):
    new = predict_stream(model, dataset, rows, config, 'cpu', **kwargs)
    old = reference_predict_stream(model, dataset, rows, config, 'cpu', **kwargs)
    assert new.keys() == old.keys()
    for key in new:
        if isinstance(new[key], np.ndarray):
            np.testing.assert_allclose(new[key], old[key], rtol=0, atol=1e-6, err_msg=key)
        else:
            assert new[key] == old[key], key


def stream(index, length=29):
    rows = [r for d in index.documents('dev') + index.documents('fit') for r in range(d.first, d.first + d.length)]
    return tuple(rows[:length])


@pytest.mark.parametrize('window', [1, 2, 3, 4, 8, 12])
@pytest.mark.parametrize('batch_size', [1, 3, 64])
def test_boundary_feature_model_deployment_policy(tmp_path, window, batch_size):
    _, index, dataset = setup_data(tmp_path)
    torch.manual_seed(window)
    model = model_for(dataset).eval()
    rows = stream(index)
    config = deployment_inference(window, batch_size)
    compare(model, dataset, rows, config)
    compare(model, dataset, rows[:window], config)
    if window > 1:
        compare(model, dataset, rows, config, collect_embeddings=True, collect_window_stats=True)
        starts = np.zeros(len(rows), int)
        starts[[0, 5, 6, 17]] = 1
        ends = np.r_[starts[1:], 1]
        compare(model, dataset, rows, config, document_starts=starts, document_ends=ends)


@pytest.mark.parametrize('merge', ['center_logits', 'uniform_probs', 'center_probs', 'central'])
def test_merges_and_anchored_tail(tmp_path, merge):
    _, index, dataset = setup_data(tmp_path)
    model = model_for(dataset).eval()
    config = InferenceConfig(window_size=8, stride=5, merge=merge, edge_ramp=2, batch_size=4, tail='anchored')
    compare(model, dataset, stream(index), config)


def test_scan_border_tokens_and_plain_models(tmp_path):
    _, index, dataset = setup_data(tmp_path)
    plain = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    backbone = dict(pretrained=False, config={
        'vocab_size': 8, 'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 4,
        'intermediate_size': 32, 'max_position_embeddings': 128,
        'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.})
    dims = dict(plain.features.dims)
    if 'pagination' in getattr(plain, 'model_dims', {}):
        dims['pagination'] = 5
    rows = stream(index)
    bordered = PageTagger(BertBackbone.create(**backbone), dims, scan_borders=True).eval()
    for data in (plain, ScanBorderDataset(plain)):
        compare(bordered, data, rows, InferenceConfig(window_size=8, stride=6, edge_ramp=1, batch_size=5))
    unbordered = PageTagger(BertBackbone.create(**backbone), dims).eval()
    compare(unbordered, plain, rows, InferenceConfig(window_size=6, stride=4, edge_ramp=1, batch_size=2))
