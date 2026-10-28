import numpy as np
import pytest

from embeddings.image.dinov2.thumbnail import original_thumbnail


def _page(height, width):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_images_up_to_256_px_are_unchanged():
    page = _page(256, 180)
    assert original_thumbnail(page) is page


@pytest.mark.parametrize('shape', [(3001, 2000), (1500, 2400), (700, 513), (1, 4000)])
def test_the_longest_edge_becomes_256_px_and_the_other_follows_python_round(shape):
    height, width = shape
    thumbnail = original_thumbnail(_page(height, width))
    # two steps, each rounding the shorter side of its own result
    step = round(min(shape) * 512 / max(shape)) if max(shape) > 512 else min(shape)
    long_side = 512 if max(shape) > 512 else max(shape)
    expected_short = max(1, round(step * 256 / long_side))
    assert max(thumbnail.shape[:2]) == 256
    assert min(thumbnail.shape[:2]) == expected_short


def test_the_thumbnail_is_the_one_the_metakat_engine_makes():
    dino = pytest.importorskip('metakat.page_type.engines.core.context.dino')
    for shape in [(3001, 2000), (1500, 2400), (700, 513), (200, 150)]:
        page = _page(*shape)
        assert np.array_equal(original_thumbnail(page), dino.original_thumbnail(page))
