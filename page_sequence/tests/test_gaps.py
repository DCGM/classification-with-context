import json

import pytest

from page_sequence.gaps import gap_fraction, write_ignore_list
from page_sequence.index import PageIndex, build_index
from page_sequence.schema import Page


def test_gap_filter_keeps_small_gaps_and_records_exclusions(tmp_path):
    orders = {'dense': [100, 101, 102], 'small_gap': [0, 1, 3, 4],
              'step_two': [0, 2, 4, 6], 'sparse': [0, 100], 'singleton': [500]}
    docs = [{'library': 'a', 'doc': doc, 'split': 'test' if doc == 'sparse' else 'fit'}
            for doc in orders]
    pages = [Page('a', doc, f'{doc}-{order}', order, db_type='TitlePage')
             for doc, indices in orders.items() for order in indices]
    annotation = Page('a', 'step_two', 'step_two-2', 2, annotated_type='TitlePage')
    path = tmp_path / 'index.sqlite'
    build_index(path, docs, pages, [annotation])
    index = PageIndex(path)
    assert {d.doc for d in index.documents('fit')} == {'dense', 'small_gap', 'singleton'}
    assert index.documents('test') == []
    assert index.db.execute('SELECT COUNT(*) FROM pages').fetchone() == (8,)
    assert index.db.execute('SELECT COUNT(*) FROM anchors').fetchone() == (8,)
    out = tmp_path / 'ignored.jsonl'
    write_ignore_list(index.db, out)
    excluded = {r['doc']: r for r in map(json.loads, out.read_text().splitlines())}
    assert set(excluded) == {'step_two', 'sparse'}
    assert excluded['step_two']['missing_positions'] == 3
    assert excluded['step_two']['gap_fraction'] == pytest.approx(3 / 7)
    assert excluded['sparse']['split'] == 'test'
    assert index.db.execute("SELECT value FROM meta WHERE key='max_gap_fraction'").fetchone() == ('0.3',)


def test_gap_threshold_is_strict_and_configurable(tmp_path):
    assert gap_fraction(7, 0, 9) == .3
    docs = [{'library': 'a', 'doc': 'x', 'split': 'fit'}]
    pages = [Page('a', 'x', str(i), i) for i in [0, 1, 2, 3, 4, 5, 9]]
    build_index(tmp_path / 'index', docs, pages, max_gap_fraction=.3)
    assert len(PageIndex(tmp_path / 'index').documents('fit')) == 1


@pytest.mark.parametrize('threshold', [-1, 2, float('nan')])
def test_invalid_gap_threshold(tmp_path, threshold):
    with pytest.raises(ValueError, match='max_gap_fraction'):
        build_index(tmp_path / 'index', [{'library': 'a', 'doc': 'x', 'split': 'fit'}],
                    [Page('a', 'x', 'p', 0)], max_gap_fraction=threshold)
