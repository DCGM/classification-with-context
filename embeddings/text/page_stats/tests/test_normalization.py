"""Bounded model inputs from stored statistics; all examples are synthetic or public fixtures."""
import json

import numpy as np
import pytest

from embeddings.text.page_stats import FEATURES, LANGUAGE_GROUPS, PageStats, PageStatsNormalization
from embeddings.text.page_stats.tests.test_page_stats import ALTO, TEXT


def rows(**values):
    count = len(next(iter(values.values())))
    full = np.zeros((count, len(FEATURES)), dtype=np.float32)
    for name, column in values.items():
        full[:, FEATURES.index(name)] = column
    return full


@pytest.mark.parametrize('grouped', [False, True])
def test_fixed_normalization_is_bounded_and_batch_independent(grouped):
    stats = PageStats()
    full = np.stack([stats.embed_text(TEXT), stats.embed_alto(ALTO), stats.embed_text('')])
    before = full.copy()
    x, names = stats.view(full, grouped=grouped, normalize=True)
    raw, raw_names = stats.view(full, grouped=grouped)
    assert names == raw_names and x.shape == raw.shape and x.dtype == np.float32
    assert np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
    np.testing.assert_array_equal(full, before)
    for i in range(len(full)):
        one, _ = stats.view(full[i], grouped=grouped, normalize=True)
        np.testing.assert_array_equal(one[0], x[i])
    np.testing.assert_array_equal(x[:, names.index('has_ocr_xml')], [0, 1, 0])
    np.testing.assert_array_equal(x[:, names.index('empty')], [0, 0, 1])
    # Source-missing geometry stays zero, not an imputed mean or a negative standardized value.
    assert x[0, names.index('page_aspect')] == 0


def test_known_scales_ratios_and_no_double_logging():
    stats = PageStats()
    full = rows(**{'kw:obsah': [0, 1, 1000], 'log_chars': [0, np.log(2), np.log(1001)],
                   'mean_line_len': [0, 1, 1000], 'line_height_med': [0, .01, .1],
                   'digit_ratio': [0, .25, 1.5]})
    x, names = stats.view(full, normalize=True, include=['kw:obsah', 'log_chars', 'mean_line_len',
                                                      'line_height_med', 'digit_ratio'])
    for name in ('kw:obsah', 'log_chars', 'mean_line_len'):
        np.testing.assert_allclose(x[:, names.index(name)], [0, .5, np.log(1001) / (np.log(1001) + np.log(2))])
    np.testing.assert_allclose(x[:, names.index('line_height_med')], [0, .5, 10 / 11])
    np.testing.assert_allclose(x[:, names.index('digit_ratio')], [0, .25, 1])


@pytest.mark.parametrize('grouped', [False, True])
def test_fitted_scales_ignore_zeros_and_round_trip_with_subsets(grouped):
    stats = PageStats({'contents': ['obsah']})
    full = rows(**{'kw:obsah': [0] * 99 + [100], 'line_height_med': [0] * 99 + [.002],
                   'mean_line_len': [0] * 99 + [80]})
    normalizer = stats.fit_normalization(full, grouped=grouped)
    saved = json.loads(json.dumps(normalizer.to_dict()))
    loaded = PageStatsNormalization.from_dict(saved)
    name = 'kw_contents' if grouped else 'kw:obsah'
    selected = [name, 'line_height_med', 'mean_line_len', 'kw:rejstrik'] if not grouped else [name, 'line_height_med', 'mean_line_len']
    x, names = stats.view(full[-1], grouped=grouped, include=selected, normalize=loaded)
    np.testing.assert_allclose(x[0, [names.index(n) for n in (name, 'line_height_med', 'mean_line_len')]], .5)
    # An unseen keyword uses a finite fixed fallback rather than a zero fitted denominator.
    query = rows(**{'kw:rejstrik': [1], 'kw:obsah': [100]})
    if not grouped:
        unseen, _ = stats.view(query, include=['kw:rejstrik'], normalize=loaded)
        assert unseen[0, 0] == pytest.approx(.5)
    original, _ = stats.view(query, grouped=grouped, normalize=normalizer)
    restored, _ = stats.view(query, grouped=grouped, normalize=loaded)
    np.testing.assert_array_equal(original, restored)
    zeros, _ = stats.view(np.zeros_like(full[:1]), grouped=grouped, normalize=loaded)
    assert not zeros.any()


def test_grouped_flags_and_custom_language_groups():
    stats = PageStats(groups=LANGUAGE_GROUPS)
    full = stats.embed_text(TEXT)
    normalizer = stats.fit_normalization(full, grouped=True, include=['head_*', 'kw_*'])
    raw, names = stats.view(full, grouped=True, include=['head_*', 'kw_*'])
    normalized, _ = stats.view(full, grouped=True, include=['head_*', 'kw_*'], normalize=normalizer)
    flags = [i for i, name in enumerate(names) if name.startswith('head_')]
    np.testing.assert_array_equal(normalized[:, flags], raw[:, flags])
    with pytest.raises(ValueError, match='different grouping'):
        PageStats().view(full, grouped=True, normalize=normalizer)
    with pytest.raises(ValueError, match='different grouping'):
        stats.view(full, normalize=normalizer)
    # Identical output names with different underlying stems must not silently reuse scales.
    other = PageStats({'same': ['obsah']})
    fitted = other.fit_normalization(full, grouped=True)
    with pytest.raises(ValueError, match='different grouping'):
        PageStats({'same': ['rejstrik']}).view(full, grouped=True, normalize=fitted)


def test_normalization_validation_and_empty_views():
    stats = PageStats()
    full = stats.embed_text(TEXT)
    fitted = stats.fit_normalization(full, include=['mean_line_len'])
    with pytest.raises(ValueError, match='features missing'):
        stats.view(full, normalize=fitted)
    with pytest.raises(ValueError, match='log_counts=True'):
        stats.view(full, log_counts=False, normalize=True)
    with pytest.raises(TypeError, match='normalize must'):
        stats.view(full, normalize='yes')
    with pytest.raises(ValueError, match='empty training sample'):
        stats.fit_normalization(np.zeros((0, len(FEATURES))))
    for grouped in (False, True):
        x, names = stats.view(np.zeros((0, len(FEATURES))), grouped=grouped, normalize=True)
        assert x.shape == (0, len(names))
        x, names = stats.view(full, grouped=grouped, include=[], normalize=True)
        assert x.shape == (1, 0) and names == ()
    state = fitted.to_dict()
    state['page_stats_version'] = 'unknown'
    with pytest.raises(ValueError, match='version'):
        PageStatsNormalization.from_dict(state)
    for scale in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match='invalid normalization'):
            PageStatsNormalization({'mean_line_len': {'transform': 'log_soft', 'scale': scale}})


@pytest.mark.parametrize('value', [-1, np.nan, np.inf])
def test_invalid_measurements_fail_explicitly(value):
    stats = PageStats()
    full = rows(mean_line_len=[value])
    with pytest.raises(ValueError, match='finite nonnegative'):
        stats.view(full, include=['mean_line_len'], normalize=True)
    with pytest.raises(ValueError, match='finite nonnegative'):
        stats.fit_normalization(full, include=['mean_line_len'])


def test_extreme_values_remain_finite_and_monotone():
    full = rows(mean_line_len=[0, 1e-8, 1, 100, 1e30], log_chars=[0, 1e-8, 1, 100, 1e30])
    x, _ = PageStats().view(full, include=['mean_line_len', 'log_chars'], normalize=True)
    assert np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all()
    assert (np.diff(x, axis=0) >= 0).all()


def test_ocr_coordinate_fractions_outside_page_are_clipped():
    full = rows(text_top=[-.02, .2, 1.1], text_left=[-.1, .5, 2.])
    original = full.copy()
    for normalizer in (True, PageStats().fit_normalization(full)):
        x, names = PageStats().view(full, normalize=normalizer)
        np.testing.assert_allclose(x[:, names.index('text_top')], [0, .2, 1])
        np.testing.assert_allclose(x[:, names.index('text_left')], [0, .5, 1])
    np.testing.assert_array_equal(full, original)
    with pytest.raises(ValueError, match='finite nonnegative'):
        PageStats().view(rows(text_top=[np.nan]), normalize=True)


@pytest.mark.parametrize('grouped', [False, True])
def test_cli_fixed_and_saved_normalization(tmp_path, capsys, grouped):
    from embeddings.text.page_stats.page_stats import main
    text = tmp_path / 'page.txt'
    text.write_text(TEXT)
    state = tmp_path / 'normalization.json'
    state.write_text(json.dumps(PageStats().fit_normalization(PageStats().embed_text(TEXT), grouped=grouped).to_dict()))
    base = [str(text), '--text'] + (['--grouped'] if grouped else [])
    for options in (['--normalize'], ['--normalization', str(state)]):
        main(base + options)
        output = capsys.readouterr().out
        assert output
        assert all(0 <= float(line.split('\t')[1]) <= 1 for line in output.splitlines())
    # A saved feature subset should work without requesting additional unfitted CLI features.
    selected = PageStats().fit_normalization(PageStats().embed_text(TEXT), grouped=grouped,
                                            include=['log_chars'])
    state.write_text(json.dumps(selected.to_dict()))
    main(base + ['--normalization', str(state)])
    assert capsys.readouterr().out.split('\t')[0] == 'log_chars'
