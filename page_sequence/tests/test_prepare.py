import csv
import sqlite3

from page_sequence.index import PageIndex
from page_sequence.manifests import load_streams
from page_sequence.prepare import development_group, prepare


def test_corpus_preparation_quarantines_and_holds_out_documents(tmp_path):
    source, audit, output = (tmp_path / x for x in ('source', 'audit', 'output'))
    source.mkdir()
    audit.mkdir()
    rows = []
    counts = []
    for i in range(40):
        split = 'tst' if i == 39 else 'trn'
        doc = f'doc-{i}'
        for j in range(4):
            rows.append((split, ['lib', '', '', doc, f'page-{i}-{j}', 'NormalPage',
                                 'NormalPage' if j == 0 else '', 'left', str(j), '', '',
                                 'public', f'public.256/{doc}/{j}.jpg']))
        counts.append((split, 'lib', doc, 4, 0, 3, 1))
    rows.append(('trn', ['lib', '', '', 'unordered', 'missing', '', '', '', '', '', '', 'public', '']))
    counts.append(('trn', 'lib', 'gappy', 2, 0, 4, 0))
    for j in (0, 4):
        rows.append(('trn', ['lib', '', '', 'gappy', f'gappy-{j}', '', '', '', str(j), '', '', 'public', '']))
    counts.append(('trn', 'lib', 'ambiguous', 4, 0, 3, 0))
    for j, order in enumerate((0, 1, 1, 3)):
        rows.append(('trn', ['lib', '', '', 'ambiguous', f'ambiguous-{j}', '', '', '', str(order), '', '', 'public', '']))
    for split in ('trn', 'tst'):
        with (source / f'db.256.final.pruned.{split}_docs.csv').open('w') as f:
            csv.writer(f).writerows(r for s, r in rows if s == split)
        with (source / f'annotated.{split}.256.final.csv').open('w') as f:
            csv.writer(f).writerows(r for s, r in rows if s == split and r[6])
    with sqlite3.connect(audit / 'counts.sqlite') as db:
        db.execute('CREATE TABLE counts(split,library,doc,n,lo,hi,gold)')
        db.executemany('INSERT INTO counts VALUES (?,?,?,?,?,?,?)', counts)
    with sqlite3.connect(audit / 'missing-order-v2.sqlite') as db:
        db.execute('CREATE TABLE pages(library,doc)')
        db.execute("INSERT INTO pages VALUES ('lib','unordered')")
    report = prepare(source, audit, output)
    index = PageIndex(output / 'index.sqlite')
    assert index.db.execute('PRAGMA auto_vacuum').fetchone()[0] == 2
    assert index.db.execute('PRAGMA freelist_count').fetchone()[0] == 0
    assert report['pages']['test'] == 4
    assert report['monitor_pages'] > 0
    assert len(load_streams(output / 'dev.jsonl', index, 'dev')) > 0
    assert not index.db.execute("SELECT 1 FROM documents WHERE doc IN ('unordered','gappy','ambiguous')").fetchone()
    assert report['ambiguous_documents'] == 1
    assert {r[0] for r in index.db.execute('SELECT copy_group FROM documents GROUP BY copy_group HAVING COUNT(DISTINCT split)>1')} == set()
    assert development_group('same-document', 17, .15) == development_group('same-document', 17, .15)
