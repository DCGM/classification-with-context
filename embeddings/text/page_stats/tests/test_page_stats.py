import numpy as np
import pytest

from embeddings.text.page_stats import FEATURES, GROUPED_FEATURES, KEYWORD_GROUPS, STEMS, PageStats

ALTO = b'''<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#"><Layout>
<Page WIDTH="1000" HEIGHT="2000"><PrintSpace>
<TextBlock><TextLine HPOS="300" VPOS="100" WIDTH="400" HEIGHT="80">
<String CONTENT="OBSAH" WIDTH="400" WC="0.9"/></TextLine></TextBlock>
<TextBlock><TextLine HPOS="100" VPOS="360" WIDTH="800" HEIGHT="40">
<String CONTENT="Rejstrik" WIDTH="300" WC="0.2"/><String CONTENT="12" WIDTH="30" WC="1"/></TextLine>
<TextLine HPOS="100" VPOS="300" WIDTH="800" HEIGHT="40">
<String CONTENT="Kapitola" WIDTH="300" WC="1"/><String CONTENT="5" WIDTH="20" WC="1"/></TextLine>
</TextBlock></PrintSpace></Page></Layout></alto>'''
TEXT = ('PŘEDMLUVA\nNapsal Jan\nÚvod ........ 5\nISBN 80-86286-64-9\nObsahem je rejstřík\n'
        'Cena 25 Kčs\nMěřítko 1 : 25 000\n')


@pytest.fixture(scope='module')
def stats():
    return PageStats()


def test_full_vector_counts_hits_per_stem(stats):
    full = dict(zip(FEATURES, stats.embed_text(TEXT)))
    assert len(FEATURES) == 3 * len(STEMS) + 3 * 8 + 22 + 32 and len(set(FEATURES)) == len(FEATURES)
    assert full['head_kw:predmluv'] == 1 and full['head_kw:predml'] == 1 and full['kw:napsal'] == 1
    assert full['kw:obsah'] == 1 and full['head_kw:obsah'] == 0  # 'Obsahem' (word start), not in the header
    assert full['kw:rejstrik'] == 1 and full['kw:rejstr'] == 1  # nested stems both count
    assert full['foot_kw:cena '] == 1 and full['pat:isbn'] == 1 and full['foot_pat:scale'] == 1
    assert full['has_ocr_xml'] == 0  # plain text: no ALTO features


def test_alto_lines_are_read_top_to_bottom(stats):
    full = dict(zip(FEATURES, stats.embed_alto(ALTO)))
    assert full['head_kw:obsah'] == 1 and full['has_ocr_xml'] == 1
    assert full['page_aspect'] == pytest.approx(0.5) and full['conf_low'] == pytest.approx(1 / 5)
    # the text features see the lines top to bottom, although the XML has Rejstrik before Kapitola
    n_text = len(FEATURES) - 32
    np.testing.assert_array_equal(stats.embed_alto(ALTO)[:n_text], stats.embed_text('OBSAH\nKapitola 5\nRejstrik 12')[:n_text])


def test_grouped_view_and_selection(stats):
    full = stats.embed([TEXT, TEXT], kind='text')
    x, names = stats.view(full, grouped=True)
    assert names == GROUPED_FEATURES and x.shape == (2, len(GROUPED_FEATURES))
    g = dict(zip(names, x[0]))
    assert g['head_preface'] == 1 and g['kw_preface'] == pytest.approx(np.log1p(3))  # 'predmluv', 'predml', 'uvod'
    assert g['kw_index'] == pytest.approx(np.log1p(2)) and g['foot_pat_scale'] == 1
    x, names = stats.view(full, include=['alto_lines', 'kw:obsah'], exclude=['has_ocr_xml'])
    assert 'kw:obsah' in names and 'has_ocr_xml' not in names and len(names) == 27
    assert x[0, names.index('kw:obsah')] == pytest.approx(np.log1p(1))
    x, names = stats.view(full, grouped=True, include=['head_*'], exclude=['head_toc'])
    assert len(names) == len(KEYWORD_GROUPS) - 1 + 8  # head_<group> and head_pat_<pattern>
    raw, _ = stats.view(full.astype(np.float16), log_counts=False)  # e.g. read back from an LMDB
    assert raw[0, FEATURES.index('kw:obsah')] == 1


def test_custom_groups_regroup_stored_vectors(stats):
    full = stats.embed_text(TEXT)
    x, names = PageStats({'czech_preface': ['predmluv', 'uvod']}).view(full, grouped=True, include=['kw_*'])
    assert names == ('kw_czech_preface',) and x[0, 0] == pytest.approx(np.log1p(2))
    with pytest.raises(ValueError):
        PageStats({'new': ['notastem']})
    with pytest.raises(ValueError):
        stats.view(np.zeros((1, 10)))


def test_old_groups_reproduce_the_experiment_vector():
    """Regrouped with the keyword groups of the experiments, the stored vector gives the cues + ALTO features
    evaluated in embedding_probe (probe_text_cues --alto-lmdb): every old stem is still extracted. Only the
    hyphenated stems, which could never match, are gone."""
    from embedding_probe.ocr_layout import LAYOUT_XML_FEATURES, layout_vector, ocr_text_lines
    from embedding_probe.text_cues import CUE_FEATURES, KEYWORDS as OLD_KEYWORDS, cue_vector
    missing = {s for stems in OLD_KEYWORDS.values() for s in stems} - set(STEMS)
    assert missing == {'avant-propos', 'x-rite'}
    old_stats = PageStats({g: [s for s in stems if s not in missing] for g, stems in OLD_KEYWORDS.items()})
    for alto in (ALTO, ALTO.replace(b'OBSAH', b'Seznam map a Vorwort')):
        old = np.array(cue_vector('\n'.join(ocr_text_lines(alto))) + layout_vector(alto), dtype=np.float32)
        new, names = old_stats.view(old_stats.embed_alto(alto), grouped=True)
        old = dict(zip(CUE_FEATURES + LAYOUT_XML_FEATURES, old))  # compared by name: short_words moved
        new = {n: v for n, v in zip(names, new[0]) if not n.startswith('head_pat_')}  # new in version 1
        assert set(new) == set(old)
        if alto != ALTO:  # short_words now counts the text's tokens, the old one ALTO String elements
            old.pop('short_words')
        np.testing.assert_allclose([new[n] for n in old], list(old.values()), rtol=1e-6)


def test_readme_lists_the_current_keywords():
    from pathlib import Path
    from embeddings.text.page_stats.page_stats import keyword_table
    readme = (Path(__file__).parents[1] / 'README.md').read_text(encoding='utf-8')
    assert keyword_table() in readme, 'regenerate the README keyword tables (--keyword-table)'


def test_sparse_storage_round_trip(stats):
    from embeddings.text.page_stats.page_stats import from_sparse, to_sparse
    full = stats.embed_alto(ALTO)
    data = to_sparse(full)
    assert len(data) == 4 * np.count_nonzero(full)
    np.testing.assert_allclose(from_sparse(data), full.astype(np.float16), rtol=0)
    np.testing.assert_array_equal(stats.decode([data, to_sparse(np.zeros(len(FEATURES)))])[1], 0)


def test_trie_matches_the_word_start_regex():
    import re
    from embeddings.text.page_stats.page_stats import normalize, stem_counts
    text = ' ' + normalize('Rejstřík jmen; obsahuje viz str. 5, kap. 3 — Vorwort; x-rite; 2rejstr; místo čti; '
                           'Содержание; tabulky tabulek; seznam map a plany\nINDEX alphabétique') + ' '
    expected = [len(re.findall(r'(?<![^\W\d_])' + re.escape(s), text)) for s in STEMS]
    assert stem_counts(text) == expected and sum(expected) > 10
