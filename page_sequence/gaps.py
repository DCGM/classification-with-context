"""Simple document gap screening, independent of features and model architecture."""
import json
import math
from pathlib import Path


DEFAULT_MAX_GAP_FRACTION = 0.30


def gap_fraction(count, first, last):
    """Missing fraction inside the observed index range; unknown outer gaps are ignored."""
    if count < 1 or last - first + 1 < count:
        raise ValueError('invalid unique page index range')
    span = last - first + 1
    return (span - count) / span


def filter_documents(db, max_gap_fraction=DEFAULT_MAX_GAP_FRACTION):
    """Audit every document, then remove excluded documents from staging and the manifest.

    Run after identity/split checks. Keep the audit in the published index so exclusions
    (including development/test exclusions) remain visible and reproducible.
    """
    if not math.isfinite(max_gap_fraction) or not 0 <= max_gap_fraction <= 1:
        raise ValueError('max_gap_fraction must be between 0 and 1')
    db.executescript('''
        CREATE TABLE document_gap_audit(
            library TEXT, doc TEXT, split TEXT, observed_pages INTEGER,
            first_order INTEGER, last_order INTEGER, missing_positions INTEGER,
            gap_fraction REAL, excluded INTEGER, reason TEXT,
            PRIMARY KEY(library,doc));
    ''')
    rows = db.execute('SELECT d.library,d.doc,d.split,COUNT(s.page),MIN(s.ord),MAX(s.ord) '
                      'FROM documents d LEFT JOIN staging s USING(library,doc) GROUP BY d.id')
    pending = []
    for library, doc, split, count, first, last in rows:
        if not count:
            raise ValueError('empty document in split manifest')
        fraction = gap_fraction(count, first, last)
        excluded = fraction > max_gap_fraction
        pending.append((library, doc, split, count, first, last, last - first + 1 - count,
                        fraction, int(excluded), 'excessive_index_gaps' if excluded else ''))
        if len(pending) >= 10000:
            db.executemany('INSERT INTO document_gap_audit VALUES (?,?,?,?,?,?,?,?,?,?)', pending)
            pending.clear()
    db.executemany('INSERT INTO document_gap_audit VALUES (?,?,?,?,?,?,?,?,?,?)', pending)
    db.execute('CREATE INDEX gap_excluded ON document_gap_audit(excluded)')
    for table in ('staging', 'documents'):
        db.execute(f'DELETE FROM {table} WHERE (library,doc) IN '
                   '(SELECT library,doc FROM document_gap_audit WHERE excluded=1)')
    if not db.execute('SELECT 1 FROM documents LIMIT 1').fetchone():
        raise ValueError('gap filter excluded every document')


def write_ignore_list(db, path):
    """Export machine-readable exclusions with their measurements and original splits."""
    cursor = db.execute('SELECT * FROM document_gap_audit WHERE excluded=1 ORDER BY library,doc')
    names = [column[0] for column in cursor.description]
    with Path(path).open('x', encoding='utf-8') as f:
        for row in cursor:
            f.write(json.dumps(dict(zip(names, row))) + '\n')
