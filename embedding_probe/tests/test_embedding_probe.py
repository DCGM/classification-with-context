import json

import lmdb
import numpy as np
import polars as pl
import pytest

from embedding_probe.build_pages import windows
from embedding_probe.common import LABEL2ID, PAGE_TYPES, evaluate, image_key, label_ids, window_completeness, window_ids
from embedding_probe.extract_vectors import Vectors, read_vectors
from embedding_probe.probe_embeddings import (class_weights, fit_linear, knn_features, knn_predict, linear_features,
                                              predict_linear, standardized, training_labels)
from embedding_probe.text_baselines import (LAYOUT_FEATURES, layout_features, presence_flags, slot_rows,
                                            window_dense, window_sparse)


@pytest.fixture
def pages():
    """Two documents: d1 with pages p0, p2, p4 (orders 0, 2, 10, listed shuffled), d2 with the single page q0."""
    context = pl.DataFrame({'library': ['a'] * 4, 'doc': ['d1', 'd1', 'd1', 'd2'],
                            'page': ['p2', 'p0', 'p4', 'q0'], 'order': ['2', '0', '10', '0'],
                            'path': ['public.256/a/d1.images/p2.jpg', 'public.256/a/d1.images/p0.jpg',
                                     'public.256/a/d1.images/p4.jpg', 'private.256/a/d2.images/q0.jpg']})
    return windows(context, 2).sort('page')


def test_label_space_is_metakat_classes_plus_colophon():
    assert len(PAGE_TYPES) == 39 and 'Colophon' in LABEL2ID and list(PAGE_TYPES) == sorted(PAGE_TYPES)
    assert label_ids(['Abstract', '', None, 'introduction']).tolist() == [0, -1, -1, -1]


def test_evaluate_weighted_over_all_classes_and_unknown_predictions_are_wrong():
    result = evaluate(['Blank', 'Blank', 'Index', 'Index'], ['Blank', '', 'Index', 'Blank'])
    assert result['accuracy'] == pytest.approx(0.5)
    # Blank: P 1/2, R 1/2, F1 1/2; Index: P 1, R 1/2, F1 2/3; equal support
    assert result['weighted_fscore'] == pytest.approx((0.5 + 2 / 3) / 2)
    assert set(result['per_class']) == {'Blank', 'Index'}


def test_windows_follow_order_within_documents(pages):
    rows = {r['page']: r for r in pages.to_dicts()}
    assert [rows[p]['rank'] for p in ('p0', 'p2', 'p4')] == [0, 1, 2]  # numeric order, not text order
    assert rows['p2']['window_page'] == [None, 'p0', 'p2', 'p4', None]
    assert rows['p0']['window_page'] == [None, None, 'p0', 'p2', 'p4']
    assert rows['q0']['window_page'] == [None, None, 'q0', None, None]  # never crosses into d1


def test_window_ids_and_completeness(pages):
    assert window_ids(pages, 'text', 0) == [['a_p0', 'a_p2', 'a_p4', 'a_q0']]
    left, center, right = window_ids(pages, 'image', 1)
    assert left == [None, 'public.256/a/d1.images/p0.jpg', 'public.256/a/d1.images/p2.jpg', None]
    assert center[3] == 'private.256/a/d2.images/q0.jpg' and right[2] is None
    assert window_completeness(pages, 0).tolist() == ['full'] * 4
    assert window_completeness(pages, 1).tolist() == ['left cut', 'full', 'right cut', 'both cut']
    assert window_completeness(pages, 2).tolist() == ['left cut', 'both cut', 'right cut', 'both cut']
    with pytest.raises(AssertionError):
        window_ids(pages, 'text', 3)


def test_image_key_strips_the_root():
    assert image_key('private.256/knav/d.images/p.jpg') == ('private', 'knav/d.images/p.jpg')


def _lmdb(path, items):
    env = lmdb.open(str(path))
    with env.begin(write=True) as txn:
        for key, value in items.items():
            txn.put(key.encode(), np.full(2, value, dtype=np.float16).tobytes())
    return env


def test_read_vectors_text_tries_public_then_private_and_image_uses_the_root(tmp_path):
    envs = {'public': _lmdb(tmp_path / 'pub', {'mzk_a': 1.0}),
            'private': _lmdb(tmp_path / 'priv', {'mzk_a': 2.0, 'mzk_b': 3.0})}
    found = read_vectors(envs, ['mzk_a', 'mzk_b', 'mzk_missing'], 'text', 'float16', threads=2)
    assert found['mzk_a'][0] == 1 and found['mzk_a'][1].tolist() == [1.0, 1.0]  # public wins
    assert found['mzk_b'][0] == 2 and found['mzk_b'][1].tolist() == [3.0, 3.0]
    assert set(found) == {'mzk_a', 'mzk_b'}
    found = read_vectors(envs, ['private.256/mzk_b', 'public.256/mzk_b'], 'image', 'float16', threads=2)
    assert set(found) == {'private.256/mzk_b'}


def _vectors(tmp_path, name, ids, values):
    (tmp_path / 'vectors').mkdir(exist_ok=True)
    np.savez(tmp_path / 'vectors' / f'{name}.npz', ids=np.array(ids), meta=json.dumps({'dim': 2}),
             vectors=np.array(values, dtype=np.float16), source=np.ones(len(ids), dtype=np.uint8))
    return Vectors(tmp_path, name)


def test_window_features_zero_empty_slots_and_flag_them(tmp_path, pages):
    vectors = _vectors(tmp_path, 'mmBERT', ['a_p0', 'a_p2', 'a_p4', 'a_q0'], [[1, 0], [0, 1], [1, 1], [2, 0]])
    fit = np.ones(4, dtype=bool)
    x0 = linear_features(pages, [vectors], 0, fit)
    assert x0.shape == (4, 2)
    x1 = linear_features(pages, [vectors], 1, fit)
    assert x1.shape == (4, 3 * 2 + 2)  # three slots, two presence flags
    assert x1[0, 0:2].tolist() == [0, 0] and x1[0, 6] == -0.5 and x1[0, 7] == 0.5  # p0: nothing before
    assert x1[3, 6:].tolist() == [-0.5, -0.5]  # q0: alone in its document
    knn = knn_features(pages, vectors, 1)
    assert np.allclose(np.linalg.norm(knn, axis=1), 1) and knn[3, :2].tolist() == [0, 0]


def test_standardized_zeroes_missing_rows():
    x = np.array([[1.0], [3.0], [100.0]], dtype=np.float32)
    result = standardized(x, np.array([True, True, False]), np.array([True, True, True]))
    assert result[:, 0].tolist() == [-1.0, 1.0, 0.0]


def test_class_weights_balance_present_classes():
    weights = class_weights(np.array([0, 0, 0, 2]), 4)
    assert weights.tolist() == pytest.approx([4 / 6, 0, 2, 0])


def test_linear_probe_and_knn_separate_easy_classes():
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8, dtype=np.float32) * 5
    y = np.repeat(np.array([LABEL2ID['Blank'], LABEL2ID['Index'], LABEL2ID['Map']]), 40)
    x = np.repeat(centers, 40, axis=0) + rng.normal(size=(120, 8)).astype(np.float32)
    model = fit_linear(x, y, 1e-3, 'cpu')
    assert (predict_linear(model, x) == y).mean() > 0.95
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert (knn_predict(xn, y, xn, k=5) == y).mean() > 0.95


def test_text_window_features():
    import scipy.sparse as sp
    rows = slot_rows([['k1', None], ['k0', 'k1'], [None, 'k9']], {'k0': 0, 'k1': 1})
    assert [r.tolist() for r in rows] == [[1, -1], [0, 1], [-1, -1]]
    features = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert window_dense(features, rows).tolist() == [[3, 4, 1, 2, 0, 0], [0, 0, 3, 4, 0, 0]]
    assert window_sparse(sp.csr_matrix(features), rows).toarray().tolist() == window_dense(features, rows).tolist()
    assert presence_flags(rows, 1).tolist() == [[1, 0], [0, 0]]
    assert presence_flags(rows[1:2], 0).shape == (2, 0)


def test_layout_features():
    assert len(layout_features('')) == len(LAYOUT_FEATURES)
    toc = dict(zip(LAYOUT_FEATURES, layout_features('OBSAH\nÚvod ........ 5\nKapitola\n12\nIV.\n.\n')))
    assert toc['number_only_lines'] == pytest.approx(2 / 6)
    assert toc['dot_leader_lines'] == pytest.approx(1 / 6)
    assert dict(zip(LAYOUT_FEATURES, layout_features(' \n\n')))['empty'] == 1.0


def test_text_cues_spot_headings_patterns_and_word_starts():
    from embedding_probe.text_cues import CUE_FEATURES, cue_vector
    cues = dict(zip(CUE_FEATURES, cue_vector('PŘEDMLUVA\nNapsal Jan\nÚvod ........ 5\nISBN 80-86286-64-9\n'
                                               'Cena 25 Kčs\nMěřítko 1 : 25 000\n')))
    assert len(cue_vector('')) == len(CUE_FEATURES)
    assert cues['head_preface'] == 1 and cues['head_toc'] == 0
    assert cues['pat_isbn'] > 0 and cues['pat_price'] > 0 and cues['pat_scale'] > 0 and cues['pat_leader_number'] > 0
    assert cues['kw_map'] > 0  # 'měřítko' without diacritics
    no_cues = dict(zip(CUE_FEATURES, cue_vector('Kapitola o kindexu\n')))
    assert no_cues['kw_index'] == 0  # keywords match word starts only


def test_training_labels_db_drops_pages_without_db_type_but_keeps_tst():
    pages = pl.DataFrame({'split': ['trn', 'trn', 'tst', 'tst'], 'label': ['Blank', 'Index', 'Map', 'Map'],
                          'db_type': ['NormalPage', None, None, 'Map']})
    y, trn, tst = training_labels(pages, 'db')
    assert y[0] == LABEL2ID['NormalPage'] and trn.tolist() == [True, False, False, False]
    assert tst.tolist() == [False, False, True, True]  # a tst page without a DB type is still scored
    y, trn, tst = training_labels(pages, 'annotated')
    assert y[0] == LABEL2ID['Blank'] and trn.tolist() == [True, True, False, False]
