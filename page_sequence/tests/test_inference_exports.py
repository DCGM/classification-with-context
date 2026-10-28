import numpy as np
import pytest
import torch

from page_sequence.inference import InferenceConfig, StreamMerger, context_weights, sigmoid, softmax


@pytest.mark.parametrize('strategy', ['center_logits', 'uniform_logits', 'center_probs', 'uniform_probs', 'central'])
def test_exports_merge_auxiliary_heads_embeddings_and_window_disagreement(strategy):
    config = InferenceConfig(4, 2, strategy, 2)
    merger = StreamMerger(6, 3, config, collect_embeddings=True, collect_window_stats=True)
    legacy = StreamMerger(6, 3, config)
    records = []
    for j, (lo, hi) in enumerate([(0, 4), (2, 6), (3, 6)]):
        n = hi - lo
        output = dict(type_logits=np.tile([j * 2., 1., -1.], (n, 1)),
                      placement_logits=np.tile([1., -j, 0.], (n, 1)),
                      starts_document_logits=np.full(n, j, np.float32),
                      pair_mask=np.arange(n) > 0,
                      page_start_logits=np.full(n, j - 1., np.float32),
                      page_end_logits=np.full(n, 1. - j, np.float32),
                      embeddings=np.tile([j, 2 * j + 1], (n, 1)).astype(np.float16))
        merger.add(lo, hi, output)
        legacy.add(lo, hi, {k: v for k, v in output.items()
                            if k not in ('embeddings', 'page_start_logits', 'page_end_logits')})
        weights = context_weights(lo, hi, 6, 2)
        if strategy.startswith('uniform'):
            weights[:] = 1
        records.append((lo, hi, output, weights))
    result, old = merger.finalize(), legacy.finalize()
    for key in old:
        np.testing.assert_array_equal(result[key], old[key])
    for i in range(6):
        contributing = [(x, w[i - lo]) for lo, hi, x, w in records if lo <= i < hi]
        weights = np.array([w for _, w in contributing]); weights /= weights.sum()
        if strategy == 'central':
            chosen = contributing[int(np.argmax([w for _, w in contributing]))][0]
            expected_embedding = chosen['embeddings'][0]
        else:
            expected_embedding = sum(w * x['embeddings'][0] for (x, _), w in zip(contributing, weights))
        np.testing.assert_allclose(result['embeddings'][i], expected_embedding, rtol=1e-6)
        for task in ('page_start', 'page_end'):
            values = [x[task + '_logits'][0] for x, _ in contributing]
            expected = (sigmoid(chosen[task + '_logits'][0]) if strategy == 'central' else
                        sum(w * sigmoid(v) for w, v in zip(weights, values)) if strategy.endswith('probs') else
                        sigmoid(np.dot(weights, values)))
            np.testing.assert_allclose(result[task + '_probs'][i], expected, rtol=1e-6)
        probabilities = np.array([softmax(x['type_logits'][0]) for x, _ in contributing])
        mean = (probabilities * weights[:, None]).sum(0)
        expected_js = -(mean * np.log(mean)).sum() + sum(
            w * (p * np.log(p)).sum() for w, p in zip(weights, probabilities))
        assert result['type_window_js'][i] == pytest.approx(expected_js, abs=3e-7)
        assert result['window_count'][i] == len(contributing)
    assert result['embeddings'].dtype == np.float32
    assert result['page_start_probs'].shape == result['page_end_probs'].shape == (6,)
    assert result['type_window_js'][0] == 0
    assert result['type_window_js'][3] > 0


def test_inference_exposes_prehead_vectors_and_preserves_predictions_with_padding(tmp_path):
    from page_sequence.tests.test_auxiliary_edges import fixture
    from page_sequence.models import BertBackbone, PageTagger
    from page_sequence.inference import predict_stream

    _, index, features, dataset = fixture(tmp_path)
    torch.set_num_threads(1)
    backbone = BertBackbone.create(pretrained=False, config=dict(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        vocab_size=1, max_position_embeddings=16, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    model = PageTagger(backbone, features.dims, auxiliary_edges=True).eval()
    d = index.documents('fit')[0]
    rows = tuple(range(d.first, d.first + min(10, d.length)))
    config = InferenceConfig(window_size=8, stride=6, batch_size=2)
    old = predict_stream(model, dataset, rows, config)
    hidden = []
    hook = model.backbone.register_forward_hook(lambda module, args, output: hidden.append(output.detach().clone()))
    result = predict_stream(model, dataset, rows, config, collect_embeddings=True, collect_window_stats=True)
    hook.remove()
    for key in old:
        np.testing.assert_array_equal(result[key], old[key])
    assert 'embeddings' not in old and 'type_window_js' not in old
    assert result['embeddings'].shape == (len(rows), 16)
    # The first page has a single contribution: it must match BERT's actual output.
    np.testing.assert_allclose(result['embeddings'][0], hidden[0][0, 0].numpy(), rtol=1e-6, atol=1e-7)
    assert np.isfinite(result['embeddings']).all()
    assert len(rows) > 8  # Exercises a short tail and masked padding in one batch.
    assert result['window_count'].tolist() == [1] * 6 + [2] * 2 + [1] * (len(rows) - 8)
    singleton = predict_stream(model, dataset, rows[:1], config,
                               collect_embeddings=True, collect_window_stats=True)
    assert singleton['embeddings'].shape == (1, 16)
    assert singleton['window_count'].tolist() == [1]
    assert singleton['type_window_js'].tolist() == [0]
    assert np.isnan(singleton['starts_document_probs'][0])
    assert np.isfinite(singleton['page_start_probs'][0]) and np.isfinite(singleton['page_end_probs'][0])
