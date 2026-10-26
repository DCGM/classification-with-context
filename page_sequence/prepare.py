"""Prepare a corpus experiment from metadata CSVs using bounded, disk-backed ingestion.

Uses an existing streaming gap audit. Only document metadata is retained in memory;
page rows stream from CSV into the strict SQLite builder. Never reads source text/OCR.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from .index import PageIndex, build_index
from .manifests import write_streams
from .schema import Page


def development_group(doc, seed, fraction):
    digest = hashlib.sha256(f'{seed}:{doc}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64 < fraction


def prepare(source, audit, output, seed=17, dev_fraction=.15, db_url=None, reuse_metadata=False):
    source, audit, output = map(Path, (source, audit, output))
    if not 0 < dev_fraction < 1:
        raise ValueError('development fraction must be between zero and one')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'index.sqlite').exists() or (output / 'index.sqlite.building').exists():
        raise FileExistsError('choose a fresh output directory')
    db = sqlite3.connect((audit / 'counts.sqlite').as_uri() + '?mode=ro', uri=True)
    audited_pages = db.execute('SELECT SUM(n) FROM counts').fetchone()[0]
    estimated_space = max(256 * 1024**2, audited_pages * 800)
    if shutil.disk_usage(output).free < estimated_space:
        raise OSError(f'index build requires an estimated {estimated_space / 1024**3:.1f} GiB free workspace')
    if reuse_metadata:
        documents = {}
        with (output / 'documents.jsonl').open() as f:
            for line in f:
                d = json.loads(line)
                documents[d['library'], d['doc']] = d
        return build_prepared(source, output, documents, seed, dev_fraction)
    missing = sqlite3.connect((audit / 'missing-order-v2.sqlite').as_uri() + '?mode=ro', uri=True)
    unordered = set(missing.execute('SELECT DISTINCT library,doc FROM pages'))
    test_ids = {r[0] for r in db.execute("SELECT doc FROM counts WHERE split='tst'")}
    documents, excluded = {}, []
    for split, lib, doc, count, lo, hi, gold in db.execute('SELECT * FROM counts ORDER BY library,doc'):
        reason = ('missing_page_order' if (lib, doc) in unordered else
                  'ambiguous_page_order' if count > hi - lo + 1 else
                  'excessive_index_gaps' if (hi - lo + 1 - count) / (hi - lo + 1) > .3 else
                  'test_document_copy' if split == 'trn' and doc in test_ids else '')
        if reason:
            excluded.append({'library': lib, 'doc': doc, 'source_split': split, 'reason': reason,
                             'indexed_pages': count})
            continue
        assigned = 'test' if split == 'tst' else 'dev' if development_group(doc, seed, dev_fraction) else 'fit'
        # Identical document UUIDs across libraries stay together even before page UUID audit.
        documents[lib, doc] = {'library': lib, 'doc': doc, 'split': assigned,
                               'copy_group': doc, 'expected_pages': count}
    # Entirely unordered documents are absent from counts; retain them in the exclusion report.
    recorded_exclusions = {(x['library'], x['doc']) for x in excluded}
    for lib, doc in sorted(unordered - documents.keys()):
        if (lib, doc) not in recorded_exclusions:
            excluded.append({'library': lib, 'doc': doc, 'reason': 'missing_page_order'})
    # Keep at least one fit group per annotated class. A class confined to one document
    # cannot have a representative in both fit and development.
    labels = defaultdict(set)
    for filename in ('annotated.trn.256.final.csv', 'annotated.tst.256.final.csv'):
        with (source / filename).open(newline='') as f:
            for row in csv.reader(f):
                d = documents.get((row[0], row[3]))
                if d and d['split'] != 'test':
                    labels[row[6]].add(row[3])
    force_fit = set()
    assignments = {d['doc']: d['split'] for d in documents.values()}
    for label, groups in sorted(labels.items(), key=lambda pair: len(pair[1])):
        if not any(assignments[g] == 'fit' or g in force_fit for g in groups):
            force_fit.add(min(groups))
    for d in documents.values():
        if d['doc'] in force_fit:
            d['split'] = 'fit'
    (output / 'ignored.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in excluded))
    print(json.dumps({'selected_documents': len(documents), 'excluded_documents': len(excluded)}), flush=True)
    if db_url:
        hierarchy(documents, db_url, output / 'hierarchy.sqlite')
    with (output / 'documents.jsonl').open('x') as f:
        for d in documents.values():
            f.write(json.dumps(d) + '\n')

    return build_prepared(source, output, documents, seed, dev_fraction)


def build_prepared(source, output, documents, seed, dev_fraction):
    def pages():
        count = 0
        started = time.monotonic()
        for split in ('trn', 'tst'):
            with (source / f'db.256.final.pruned.{split}_docs.csv').open(newline='') as f:
                for row in csv.reader(f):
                    if len(row) != 13:
                        raise ValueError('expected 13-column metadata CSV')
                    if (row[0], row[3]) not in documents:
                        continue
                    yield Page(row[0], row[3], row[4], int(row[8]), row[12], row[5], row[6], row[7])
                    count += 1
                    if count % 250000 == 0:
                        free = shutil.disk_usage(output).free
                        print(json.dumps({'ingested_pages': count, 'seconds': round(time.monotonic() - started),
                                          'free_GiB': round(free / 1024**3, 1)}), flush=True)
                        if free < 4 * 1024**3:
                            raise OSError('stopping index build before exhausting workspace')
    build_index(output / 'index.sqlite', documents.values(), pages(), quarantine_conflicts=True)
    return finalize_prepared(output, seed, dev_fraction)


def finalize_prepared(output, seed=17, dev_fraction=.15):
    """Publish manifests/reports from a completed index, including a recovered build."""
    output = Path(output)
    index = PageIndex(output / 'index.sqlite')
    cursor = index.db.execute('SELECT * FROM document_conflict_audit')
    names = [column[0] for column in cursor.description]
    conflict_count = 0
    with (output / 'ambiguous-documents.jsonl').open('x') as f:
        for row in cursor:
            f.write(json.dumps(dict(zip(names, row))) + '\n')
            conflict_count += 1
    # The strict builder rejects page UUID overlap across splits; do not silently train on it.
    # Freeze complete development streams and a smaller annotation-bearing monitoring suite.
    write_streams(index, 'dev', output / 'dev.full.jsonl', target_pages=2048, seed=seed)
    write_streams(index, 'test', output / 'test.jsonl', target_pages=2048, seed=seed)
    gold_docs = {r[0] for r in index.db.execute(
        "SELECT DISTINCT document FROM pages p JOIN documents d ON d.id=p.document "
        "WHERE d.split='dev' AND annotated_type!=''")}
    if not gold_docs:
        raise ValueError('no curated development documents after split/quarantine')
    dev_docs = index.documents('dev')
    gold_volumes = {(d.library, d.volume) for d in dev_docs if d.id in gold_docs and d.volume}
    sibling_counts = defaultdict(int)
    selected = set(gold_docs)
    for d in dev_docs:
        key = d.library, d.volume
        if d.id not in selected and key in gold_volumes and sibling_counts[key] < 2:
            selected.add(d.id)
            sibling_counts[key] += 1

    class MonitorIndex:
        db = index.db

        def documents(self, split):
            return [d for d in dev_docs if d.id in selected]

    write_streams(MonitorIndex(), 'dev', output / 'dev.jsonl', target_pages=2048, seed=seed)
    monitor_pages = sum(d.length for d in dev_docs if d.id in selected)
    with (output / 'dev.jsonl').open() as f:
        monitor_streams = sum(1 for _ in f)
    report = {'documents': dict(index.db.execute('SELECT split,COUNT(*) FROM documents GROUP BY split')),
              'pages': dict(index.db.execute('SELECT split,SUM(length) FROM documents GROUP BY split')),
              'monitor_pages': monitor_pages, 'monitor_streams': monitor_streams,
              'ambiguous_documents': conflict_count,
              'seed': seed, 'dev_fraction': dev_fraction,
              'split_method': 'document UUID groups; page UUID cross-split overlap fails build',
              'remaining_small_gaps': 'retain observed pages without inserting placeholders'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


def hierarchy(documents, db_url, cache):
    """Verify issue -> volume -> periodical links using metadata-only batched queries."""
    import psycopg2
    from psycopg2.extras import execute_values
    known = {}
    todo = set(documents)
    db = sqlite3.connect(cache)
    db.execute('CREATE TABLE records(library TEXT,id TEXT,kind TEXT,parent TEXT,parent_library TEXT,PRIMARY KEY(library,id))')
    with psycopg2.connect(db_url, connect_timeout=10) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as q:
            q.execute('SET statement_timeout=60000')
            for level in range(3):
                keys = sorted(todo - known.keys())
                for offset in range(0, len(keys), 4000):
                    execute_values(q, 'SELECT m.library,m.id::text,m.record_type,m.parent_id::text,m.parent_library '
                                      'FROM meta_records m JOIN (VALUES %s) AS v(library,id) '
                                      'ON m.library=v.library AND m.id=v.id::uuid',
                                   keys[offset:offset + 4000], page_size=4000)
                    rows = q.fetchall()
                    db.executemany('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?)', rows)
                    db.commit()
                    for lib, ident, kind, parent, plib in rows:
                        known[lib, ident] = kind, parent, plib or lib
                    if offset % 40000 == 0:
                        print(json.dumps({'hierarchy_level': level, 'fetched': offset,
                                          'known_records': len(known)}), flush=True)
                todo = {(plib, parent) for _, parent, plib in known.values() if parent}
    for d in documents.values():
        lib = d['library']
        kind, parent, plib = known.get((lib, d['doc']), ('', None, ''))
        if kind != 'periodicalitem' or plib != lib:
            continue
        pkind, grandparent, glib = known.get((lib, parent), ('', None, ''))
        if pkind == 'periodicalvolume':
            d['volume'] = parent
            if glib == lib and known.get((lib, grandparent), ('', None, ''))[0] == 'periodical':
                d['periodical'] = grandparent
        elif pkind == 'periodical':
            d['periodical'] = parent
    db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--dev-fraction', type=float, default=.15)
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--reuse-metadata', action='store_true', help='Reuse prepared documents/hierarchy after a failed index build')
    args = parser.parse_args()
    prepare(args.source_dir, args.audit_dir, args.output_dir, args.seed, args.dev_fraction, args.db_url, args.reuse_metadata)


if __name__ == '__main__':
    main()
