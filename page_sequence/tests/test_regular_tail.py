import numpy as np
import pytest

from page_sequence.inference import InferenceConfig, window_spans
from page_sequence.scan_borders import scan_windows


def test_regular_tail_keeps_stride_instead_of_shifting_back():
    assert InferenceConfig().tail == 'regular'
    assert window_spans(1024, 512, 448, 'regular') == [(0, 512), (448, 960), (896, 1024)]
    assert window_spans(970, 512, 448, 'regular') == [(0, 512), (448, 960), (896, 970)]
    assert window_spans(970, 512, 448, 'anchored') == [(0, 512), (448, 960), (458, 970)]
    assert window_spans(960, 512, 448, 'regular') == [(0, 512), (448, 960)]
    assert window_spans(512, 512, 448, 'regular') == [(0, 512)]


@pytest.mark.parametrize('overlap', [256, 128, 64, 32, 16])
def test_regular_tail_covers_pages_pairs_and_sets_only_true_scan_endpoints(overlap):
    stride = 512 - overlap
    for n in [1, 511, 512, 513, 960, 970, 1024, 1234, 2049]:
        tiles = scan_windows(n, 512, stride, False, tail='regular')
        pages, pairs = np.zeros(n, int), np.zeros(n, int)
        for lo, hi, start, end in tiles:
            assert start == (lo == 0) and end == (hi == n)
            pages[lo:hi] += 1
            pairs[lo + 1:hi] += 1
        assert (pages > 0).all() and (pairs[1:] > 0).all()
        assert pages.max() <= 2
        assert [lo for lo, _, _, _ in tiles] == list(range(0, tiles[-1][0] + 1, stride))
        assert sum(t[3] for t in tiles) == 1


def test_unknown_tail_policy_rejected():
    with pytest.raises(ValueError, match='tail'):
        InferenceConfig(tail='unknown')


def test_deployment_inference_scales_the_metakat_policy_to_each_window():
    from page_sequence.inference import deployment_inference
    expected = {512: (448, 48, 12), 256: (224, 24, 6), 128: (112, 12, 3), 64: (56, 6, 1),
                32: (28, 3, 0), 16: (14, 1, 0), 8: (7, 1, 0), 4: (3, 1, 0), 2: (1, 1, 0)}
    for window, (stride, ramp, ignore) in expected.items():
        config = deployment_inference(window)
        assert (config.stride, config.edge_ramp, config.edge_ignore) == (stride, ramp, ignore)
        assert (config.merge, config.edge_power, config.tail) == ('center_logits', 3, 'regular')
    single = deployment_inference(1)
    assert (single.window_size, single.stride, single.edge_ignore) == (1, 1, 0)
