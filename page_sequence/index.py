"""Disk-backed ordered metadata index and strict split audit.

Build once from context/annotation CSVs and an explicit document/split hierarchy manifest.
No implicit inference that every CSV parent is a volume. Only metadata is read.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from .schema import Document, LABEL2ID, Page, SCHEMA_VERSION

CSV_COLUMNS = ('library', 'root', 'parent', 'doc', 'page', 'db_type', 'annotated_type',
               'placement', 'order', 'pagenum', 'date', 'access', 'path')


def csv_pages(paths):
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) != len(CSV_COLUMNS):
                    raise ValueError('expected the repository 13-column metadata CSV')
                r = dict(zip(CSV_COLUMNS, row))
                try:
                    order = int(r['order'])
                except ValueError:
                    raise ValueError('missing or noninteger page order') from None
                yield Page(**{k: r[k] for k in Page.__dataclass_fields__ if k != 'order'}, order=order)


def build_index(path, documents, pages, annotations=(), *, max_gap_fraction=0.30,
                quarantine_conflicts=False):
    """Atomic build; iterables stream to SQLite. Existing indexes are never overwritten.

    documents: dicts with library, doc, split (fit/dev/test), optional volume, periodical,
    copy_group. These must already respect test exclusion and known copy groups.
    Ambiguous ordering, duplicated rows, cross-split page UUIDs/copies are fatal.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + '.building')
    if staging.exists():
        raise FileExistsError(staging)
    db = sqlite3.connect(staging)
    try:
        db.executescript('''
            PRAGMA auto_vacuum=INCREMENTAL;
            PRAGMA cache_size=-65536;
            PRAGMA temp_store=FILE;
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents(id INTEGER PRIMARY KEY, library TEXT NOT NULL, doc TEXT NOT NULL,
                split TEXT NOT NULL CHECK(split IN ('fit','dev','test')), volume TEXT NOT NULL,
                periodical TEXT NOT NULL, copy_group TEXT NOT NULL, first INTEGER, length INTEGER,
                UNIQUE(library,doc));
            CREATE TABLE staging(library TEXT, doc TEXT, page TEXT, ord INTEGER, path TEXT,
                db_type TEXT, annotated_type TEXT, placement TEXT,
                UNIQUE(library,doc,page), UNIQUE(library,doc,ord));
            CREATE TABLE annotations(library TEXT, doc TEXT, page TEXT, label TEXT,
                PRIMARY KEY(library,doc,page));
        ''')
        if quarantine_conflicts:
            db.execute('CREATE TABLE expected_pages(library TEXT,doc TEXT,n INTEGER,PRIMARY KEY(library,doc))')

        def checked_documents():
            for r in documents:
                if quarantine_conflicts:
                    if r.get('expected_pages', 0) < 1:
                        raise ValueError('quarantine requires an audited expected_pages count per document')
                    db.execute('INSERT INTO expected_pages VALUES (?,?,?)',
                               (r['library'], r['doc'], r['expected_pages']))
                yield (r['library'], r['doc'], r['split'], r.get('volume', ''),
                       r.get('periodical', ''), r.get('copy_group', '') or
                       json.dumps([r['library'], r['doc']]))

        db.executemany('INSERT INTO documents(library,doc,split,volume,periodical,copy_group) '
                       'VALUES (?,?,?,?,?,?)', checked_documents())
        if db.execute('SELECT 1 FROM documents GROUP BY copy_group HAVING COUNT(DISTINCT split)>1').fetchone():
            raise ValueError('copy group crosses splits')
        def checked_pages():
            for p in pages:
                if not p.library or not p.doc or not p.page:
                    raise ValueError('empty page/document identity')
                if p.annotated_type and p.annotated_type not in LABEL2ID:
                    raise ValueError('unsupported curated type')
                yield (p.library, p.doc, p.page, p.order, p.path, p.db_type, p.annotated_type, p.placement)
        insertion = 'INSERT OR IGNORE' if quarantine_conflicts else 'INSERT'
        db.executemany(insertion + ' INTO staging VALUES (?,?,?,?,?,?,?,?)', checked_pages())
        if quarantine_conflicts:
            # Ignoring conflicting rows is only an intermediate detection step: every
            # affected document is removed whole, never retained with an arbitrary page.
            db.executescript('''
                CREATE TABLE document_conflict_audit AS
                    SELECT e.library,e.doc,d.split,e.n AS expected_pages,COUNT(s.page) AS unique_pages,
                           'ambiguous_or_inconsistent_page_order' AS reason
                    FROM expected_pages e JOIN documents d USING(library,doc)
                    LEFT JOIN staging s USING(library,doc)
                    GROUP BY e.library,e.doc HAVING COUNT(s.page)!=e.n;
                DELETE FROM staging WHERE (library,doc) IN
                    (SELECT library,doc FROM document_conflict_audit);
                DELETE FROM documents WHERE (library,doc) IN
                    (SELECT library,doc FROM document_conflict_audit);
                DROP TABLE expected_pages;
            ''')
        def checked_annotations():
            for p in annotations:
                if p.annotated_type not in LABEL2ID:
                    raise ValueError('missing or unsupported curated type')
                yield p.library, p.doc, p.page, p.annotated_type
        db.executemany('INSERT INTO annotations VALUES (?,?,?,?)', checked_annotations())
        if db.execute('SELECT 1 FROM staging s LEFT JOIN documents d USING(library,doc) '
                      'WHERE d.id IS NULL LIMIT 1').fetchone():
            raise ValueError('context document absent from split manifest')
        if db.execute('SELECT 1 FROM annotations a LEFT JOIN staging s USING(library,doc,page) '
                      'WHERE s.page IS NULL LIMIT 1').fetchone():
            raise ValueError('annotation absent from context')
        if db.execute('SELECT 1 FROM annotations a JOIN staging s USING(library,doc,page) '
                      "WHERE s.annotated_type != '' AND s.annotated_type != a.label LIMIT 1").fetchone():
            raise ValueError('conflicting curated labels')
        # Page UUID is deliberately not library-qualified here: catches cross-library copies.
        db.execute('CREATE INDEX staging_page ON staging(page)')
        if db.execute('SELECT 1 FROM staging s JOIN documents d USING(library,doc) '
                      'GROUP BY s.page HAVING COUNT(DISTINCT d.split)>1 LIMIT 1').fetchone():
            raise ValueError('page UUID crosses splits; revise/quarantine split groups')
        from .gaps import filter_documents
        filter_documents(db, max_gap_fraction)
        db.executescript('''
            CREATE TABLE pages(id INTEGER PRIMARY KEY, document INTEGER NOT NULL, page TEXT NOT NULL,
                ord INTEGER NOT NULL, path TEXT NOT NULL, db_type TEXT NOT NULL,
                annotated_type TEXT NOT NULL, placement TEXT NOT NULL);
            INSERT INTO pages(document,page,ord,path,db_type,annotated_type,placement)
                SELECT d.id,s.page,s.ord,s.path,s.db_type,COALESCE(a.label,s.annotated_type),s.placement
                FROM staging s JOIN documents d USING(library,doc)
                LEFT JOIN annotations a USING(library,doc,page) ORDER BY d.id,s.ord;
            CREATE INDEX pages_document ON pages(document,id);
            CREATE INDEX pages_uuid ON pages(page,document);
            UPDATE documents SET first=(SELECT MIN(id) FROM pages WHERE document=documents.id),
                length=(SELECT COUNT(*) FROM pages WHERE document=documents.id);
        ''')
        if db.execute('SELECT 1 FROM documents WHERE length=0 LIMIT 1').fetchone():
            raise ValueError('empty document in split manifest')
        db.executescript('''
            CREATE TABLE anchors(source TEXT, label INTEGER, row_id INTEGER, slot INTEGER,
                PRIMARY KEY(source,label,slot));
            CREATE INDEX anchors_row ON anchors(row_id);
        ''')
        # Stream anchor rows in bounded batches; only fit pages can ever enter the training sampler.
        counts = {}
        pending = []
        normal = LABEL2ID['NormalPage']
        for row_id, weak, gold in db.execute('SELECT p.id,p.db_type,p.annotated_type FROM pages p '
                                            "JOIN documents d ON d.id=p.document WHERE d.split='fit'"):
            label = LABEL2ID.get(gold or weak)
            sources = (['annotated'] if gold else []) + (['interesting'] if label is not None and label != normal else [])
            for source in sources:
                key = source, label
                slot = counts.get(key, 0)
                counts[key] = slot + 1
                pending.append((source, label, row_id, slot))
            if len(pending) >= 10000:
                db.executemany('INSERT INTO anchors VALUES (?,?,?,?)', pending)
                pending.clear()
        db.executemany('INSERT INTO anchors VALUES (?,?,?,?)', pending)
        # Staging contains only rebuildable metadata. Avoid overwriting every discarded
        # page and journaling a corpus-sized FULL auto-vacuum at the DROP transaction.
        db.execute('PRAGMA secure_delete=OFF')
        db.executescript('DROP TABLE annotations; DROP TABLE staging;')
        reclaim_space(db)
        db.execute('INSERT INTO meta VALUES (?,?)', ('schema_version', str(SCHEMA_VERSION)))
        db.execute('INSERT INTO meta VALUES (?,?)', ('max_gap_fraction', str(max_gap_fraction)))
        db.commit()
        db.close()
        os.rename(staging, path)
    except BaseException:
        db.close()
        # Keep .building for diagnosis; never present an incomplete index as usable.
        raise


def reclaim_space(db):
    """Compact only the unpublished, rebuildable index, then check before publishing.

    SQLite incremental vacuum scans the freelist repeatedly at this corpus scale.
    Full auto-vacuum avoids that scan, but a rollback journal would require another
    index-sized allocation. The .building file is disposable until all checks pass;
    a failed/interrupted build must be rebuilt, never opened as a published index.
    """
    db.commit()
    if db.execute('PRAGMA freelist_count').fetchone()[0]:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA auto_vacuum=FULL')
        db.execute('BEGIN')
        db.execute('CREATE TABLE compaction_trigger(value INTEGER)')
        db.execute('DROP TABLE compaction_trigger')
        db.commit()
        db.execute('PRAGMA auto_vacuum=INCREMENTAL')
        db.execute('PRAGMA journal_mode=DELETE')
    if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
        raise ValueError('metadata index integrity check failed')


class PageIndex:
    """Connections open lazily per process; no page table copied into Python memory."""
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self._db = None
        self._pid = None

    @property
    def db(self):
        if self._db is None or self._pid != os.getpid():
            if self._db is not None:
                self._db.close()
            self._db = sqlite3.connect(Path(self.path).as_uri() + '?mode=ro', uri=True)
            self._pid = os.getpid()
            version = self._db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if version != (str(SCHEMA_VERSION),):
                raise ValueError('incompatible page index')
        return self._db

    def __getstate__(self):
        return {'path': self.path, '_db': None, '_pid': None}

    def documents(self, split):
        return [Document(*r) for r in self.db.execute('SELECT * FROM documents WHERE split=? ORDER BY id', (split,))]

    def documents_for_rows(self, rows):
        """Only documents present in a scan; bounded queries, no split-wide allocation."""
        records = {}
        unique = tuple(dict.fromkeys(int(row) for row in rows))
        for offset in range(0, len(unique), 512):
            chunk = unique[offset:offset + 512]
            query = ('SELECT DISTINCT d.* FROM pages p JOIN documents d ON d.id=p.document '
                     f'WHERE p.id IN ({",".join("?" for _ in chunk)})')
            for record in self.db.execute(query, chunk):
                if record[0] not in records:
                    records[record[0]] = Document(*record)
        return list(records.values())

    def pages(self, rows):
        return self.pages_with_documents(rows)[0]

    def pages_with_documents(self, rows):
        """Fetch a window in bounded queries; preserve order and duplicate occurrences.

        Document IDs accompany metadata for sampler accounting, never model features.
        No persistent page cache or corpus-sized metadata array is created.
        """
        original = tuple(rows)
        ordered = tuple(int(row) for row in original)
        unique = tuple(dict.fromkeys(ordered))
        records = {}
        for offset in range(0, len(unique), 512):
            chunk = unique[offset:offset + 512]
            placeholders = ','.join('?' for _ in chunk)
            query = ('SELECT p.id,p.document,d.library,d.doc,p.page,p.ord,p.path,p.db_type,'
                     'p.annotated_type,p.placement FROM pages p JOIN documents d ON d.id=p.document '
                     f'WHERE p.id IN ({placeholders})')
            for record in self.db.execute(query, chunk):
                records[record[0]] = record[1:]
        pages, documents = [], []
        for row, key in zip(ordered, original):
            if row not in records:
                raise KeyError(key)
            record = records[row]
            documents.append(record[0])
            pages.append(Page(*record[1:]))
        return pages, documents

    def fingerprint(self):
        h = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents', required=True, help='JSONL split/hierarchy manifest')
    parser.add_argument('--context', nargs='+', required=True)
    parser.add_argument('--annotations', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-gap-fraction', type=float, default=0.30,
                        help='Exclude documents above this internal missing-position fraction (default: 0.30)')
    parser.add_argument('--ignore-list', help='Exclusions JSONL; default: OUTPUT.ignored.jsonl')
    args = parser.parse_args()
    with open(args.documents) as f:
        build_index(args.output, (json.loads(line) for line in f if line.strip()),
                    csv_pages(args.context), csv_pages(args.annotations),
                    max_gap_fraction=args.max_gap_fraction)
    index = PageIndex(args.output)
    from .gaps import write_ignore_list
    ignore_list = args.ignore_list or args.output + '.ignored.jsonl'
    write_ignore_list(index.db, ignore_list)
    print(json.dumps({'sha256': index.fingerprint(), 'documents':
                      dict(index.db.execute('SELECT split,COUNT(*) FROM documents GROUP BY split')),
                      'excluded_documents': dict(index.db.execute(
                          'SELECT split,COUNT(*) FROM document_gap_audit WHERE excluded=1 GROUP BY split')),
                      'ignore_list': ignore_list}))


if __name__ == '__main__':
    main()
