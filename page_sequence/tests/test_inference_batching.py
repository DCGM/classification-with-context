"""The batched inference path gives exactly the predictions of the earlier per-window reads (and W1 path)."""
import numpy as np
import pytest
import torch

from page_sequence.data import collate_windows, to_device
from page_sequence.inference import InferenceConfig, StreamMerger, deployment_inference, predict_stream, window_spans
from page_sequence.precision import autocast
from page_sequence.scan_borders import ScanBorderDataset
from page_sequence.schema import Window
from page_sequence.tests.test_context_training import model_for, setup_data
from page_sequence.models import BertBackbone, PageTagger


# The implementation before the batched reads, kept verbatim as the reference.
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
    merger = StreamMerger(len(rows), model.num_types, config, collect_embeddings, collect_window_stats)
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
