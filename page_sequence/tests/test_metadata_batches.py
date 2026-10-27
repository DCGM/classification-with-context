import pytest

from page_sequence.index import PageIndex, build_index
from page_sequence.schema import Page


def test_metadata_batches_preserve_order_duplicates_and_document_ids(tmp_path):
    docs = [{'library': 'a', 'doc': d, 'split': 'fit'} for d in ('x', 'y')]
    pages = [Page('a', 'x', f'x-{i}', i, db_type='NormalPage') for i in range(600)]
    pages += [Page('a', 'y', f'y-{i}', i, placement='single') for i in range(7)]
    build_index(tmp_path / 'index', docs, pages)
    index = PageIndex(tmp_path / 'index')
    rows = (600, 1, 512, 1, 607, *range(1, 608), 600)
    statements = []
    index.db.set_trace_callback(statements.append)
    fetched, document_ids = index.pages_with_documents(rows)
    index.db.set_trace_callback(None)
    assert fetched == [pages[r - 1] for r in rows]
    assert document_ids == [1 if r <= 600 else 2 for r in rows]
    assert len([s for s in statements if s.startswith('SELECT')]) == 2
    assert index.pages(iter(rows)) == fetched
    assert index.pages_with_documents(()) == ([], [])

    statements.clear()
    index.db.set_trace_callback(statements.append)
    documents = index.documents_for_rows(iter(rows))
    index.db.set_trace_callback(None)
    assert {d.id: d for d in documents} == {d.id: d for d in index.documents('fit')}
    assert len([s for s in statements if s.startswith('SELECT')]) == 2
    assert index.documents_for_rows(()) == []
    assert [d.doc for d in index.documents_for_rows((601, 607, 601))] == ['y']


def test_metadata_batches_report_first_missing_row(tmp_path):
    build_index(tmp_path / 'index', [{'library': 'a', 'doc': 'x', 'split': 'fit'}],
                [Page('a', 'x', 'p', 0)])
    index = PageIndex(tmp_path / 'index')
    with pytest.raises(KeyError) as exc:
        index.pages((1, 9000, 8000, 1))
    assert exc.value.args == (9000,)
