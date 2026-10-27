import sqlite3

from page_sequence.document_length_statistics import catalogue_statistics, summarize


def test_document_and_page_weighted_views_differ():
    report = summarize({1: 9, 100: 1})
    assert report['documents'] == 10
    assert report['pages'] == 109
    assert report['document_quantiles']['p50'] == 1
    assert report['page_weighted_document_quantiles']['p50'] == 100
    assert report['windows']['32']['documents_fitting_percent'] == 90
    assert report['windows']['32']['pages_in_fitting_documents'] == 9
    assert report['windows']['128']['pages_in_fitting_documents_percent'] == 100


def test_catalogue_split_totals_and_read_only_input(tmp_path):
    path = tmp_path / 'catalogue.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE documents(split TEXT,length INTEGER)')
        db.executemany('INSERT INTO documents VALUES (?,?)', [('fit', 4), ('fit', 4), ('dev', 200)])
    before = path.read_bytes()
    report = catalogue_statistics(path)['groups']
    assert report['all']['documents'] == 3
    assert report['all']['pages'] == 208
    assert report['fit']['length_histogram'] == {'4': 2}
    assert report['dev']['windows']['128']['documents_fitting'] == 0
    assert sum(v['pages'] for k, v in report.items() if k != 'all') == report['all']['pages']
    assert path.read_bytes() == before
