import numpy as np
import pytest

from embeddings.text.page_stats import FINE_KEYWORD_GROUPS, PageStats, PageStatsNormalization


def test_fine_view_preserves_original_prefix_and_raw_schema():
    base, fine = PageStats(), PageStats(extra_groups=FINE_KEYWORD_GROUPS)
    raw = np.zeros((2, len(base.features)), np.float32)
    raw[0, base.features.index('kw:band ')] = 3
    raw[0, base.features.index('head_kw:band ')] = 2
    raw[1, base.features.index('foot_kw:finis')] = 1
    original, names = base.view(raw, grouped=True, normalize=True)
    extended, extra_names = fine.view(raw, grouped=True, normalize=True)
    assert len(names) == 147 and len(extra_names) == 180
    assert extra_names[:147] == names
    np.testing.assert_array_equal(extended[:, :147], original)
    assert fine.metadata() == base.metadata()
    assert extended[0, extra_names.index('kw_volume')] == pytest.approx(2 / 3)
    assert extended[0, extra_names.index('head_volume')] == 1
    assert extended[1, extra_names.index('foot_end_word')] == 1
    assert np.isfinite(extended).all() and (extended >= 0).all() and (extended <= 1).all()
    selected, selected_names = fine.view(raw, grouped=True, normalize=True, include=['head_keywords'])
    assert len(selected_names) == 34 and 'head_volume' in selected_names
    np.testing.assert_array_equal(selected, extended[:, [extra_names.index(n) for n in selected_names]])
    fitted = fine.fit_normalization(raw, grouped=True)
    restored = PageStatsNormalization.from_dict(fitted.to_dict())
    np.testing.assert_array_equal(fine.view(raw, grouped=True, normalize=fitted)[0],
                                  fine.view(raw, grouped=True, normalize=restored)[0])


def test_extra_groups_reject_unknown_stems_and_colliding_names():
    with pytest.raises(ValueError, match='extra group names'):
        PageStats(extra_groups={'title': ['band ']})
    with pytest.raises(ValueError, match='new stems need re-extraction'):
        PageStats(extra_groups={'new_group': ['not_a_stored_stem']})
    with pytest.raises(ValueError, match='must contain'):
        PageStats(extra_groups={'empty': []})
