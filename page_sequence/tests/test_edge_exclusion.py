import numpy as np
import pytest

from page_sequence.inference import InferenceConfig, StreamMerger, context_weights, window_spans


def output(n, value):
    return dict(type_logits=np.tile([value, 0.], (n, 1)),
                placement_logits=np.zeros((n, 3), np.float32),
                starts_document_logits=np.full(n, value, np.float32),
                page_start_logits=np.full(n, value, np.float32),
                page_end_logits=np.full(n, value, np.float32),
                embeddings=np.full((n, 2), value, np.float32),
                pair_mask=np.arange(n) > 0)


def test_excluded_edges_ramp_and_real_endpoints():
    # No additive floor: retain the original first-page distance of one.
    np.testing.assert_array_equal(context_weights(5, 10, 10, 4), [.25, .5, .75, 1, 1])
    np.testing.assert_array_equal(context_weights(5, 15, 20, 4, 2),
                                  [0, 0, .5, 1, 1, 1, 1, .5, 0, 0])
    np.testing.assert_array_equal(context_weights(0, 5, 5, 4, 2), np.ones(5))
    # Quarter-ramp exclusion: zero through 8, reach full weight at 32.
    weights = context_weights(5, 38, 38, 32, 8)
    np.testing.assert_array_equal(weights[:8], np.zeros(8))
    assert weights[8] == np.float32(1/24)
    assert weights[19] == .5
    assert weights[31] == 1.
    np.testing.assert_array_equal(context_weights(5, 10, 10, 4, 2), [0, 0, .5, 1, 1])


@pytest.mark.parametrize('tail', ['regular', 'anchored'])
@pytest.mark.parametrize('strategy', ['center_logits', 'center_probs', 'uniform_logits', 'uniform_probs', 'central'])
def test_all_pages_and_pairs_covered_even_at_half_overlap(tail, strategy):
    for overlap in range(2, 8):
        config = InferenceConfig(window_size=8, stride=8-overlap, merge=strategy,
                                 edge_ramp=overlap//2 + 2, tail=tail, edge_ignore=overlap//2)
        for n in range(1, 35):
            merger = StreamMerger(n, 2, config, True, True)
            for lo, hi in window_spans(n, 8, config.stride, tail):
                merger.add(lo, hi, output(hi-lo, 2.))
            result = merger.finalize()
            assert np.isfinite(result['embeddings']).all()
            assert (result['window_count'] > 0).all()
            assert (result['pair_weight'][1:] > 0).all()
            np.testing.assert_allclose(result['embeddings'], 2.)
            np.testing.assert_allclose(result['type_window_js'], 0., atol=3e-7)


def test_discarded_predictions_cannot_affect_pages_embeddings_aux_or_disagreement():
    config = InferenceConfig(window_size=8, stride=4, edge_ramp=4, edge_ignore=2)
    merger = StreamMerger(12, 2, config, True, True)
    first, last = output(8, 2.), output(8, 2.)
    # Overlap [4,8): first window retains 4,5; last retains 6,7.
    for values, positions in ((first, [6, 7]), (last, [0, 1])):
        for key in ('type_logits', 'page_start_logits', 'page_end_logits', 'embeddings'):
            values[key][positions] = 1000
    first['starts_document_logits'][7] = 1000  # Both endpoints discarded.
    last['starts_document_logits'][1] = 1000
    merger.add(0, 8, first)
    merger.add(4, 12, last)
    result = merger.finalize()
    np.testing.assert_allclose(result['embeddings'], 2.)
    np.testing.assert_allclose(result['type_probs'][:, 0], 1/(1+np.exp(-2.)))
    np.testing.assert_allclose(result['page_start_probs'], 1/(1+np.exp(-2.)))
    np.testing.assert_allclose(result['starts_document_probs'][1:], 1/(1+np.exp(-2.)))
    np.testing.assert_array_equal(result['window_count'], np.ones(12))
    np.testing.assert_array_equal(result['type_window_js'], np.zeros(12))


def test_reject_trimming_that_leaves_coverage_holes():
    with pytest.raises(ValueError, match='half the overlap'):
        InferenceConfig(window_size=8, stride=5, edge_ignore=2)
