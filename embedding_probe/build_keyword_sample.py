"""Development tool: a varied sample of PUBLIC pages from documents not used so far, to look for missing
keyword stems of the page statistics (embeddings/text/page_stats).

From the library metadata DB: per DB page type a random set of candidate pages (all pages of rare types),
their document languages (the 'Language' of the nearest ancestor that has one) and the top-level record
type. Pages are kept only if their text is in the PUBLIC text LMDB (the private one is never opened) and
their document is neither in the experiment pages (pages.parquet) nor in the dev sample; then up to
--per-cell pages per (page type, language), at most --per-doc per document and type, so that each
language and many documents are covered.

Writes <work-dir>/dev/keyword_sample.jsonl (key, library, page, db_type, languages, top_type, top) --
page ids and DB metadata only. Runs in the db_text_dump venv (psycopg2, lmdb); the used documents come
from <work-dir>/dev/used_docs.txt, written by --write-used-docs in a venv with polars.

Example:
    python -m embedding_probe.build_keyword_sample --write-used-docs     # metakat venv
    python -m embedding_probe.build_keyword_sample                       # db_text_dump venv
"""
import argparse
import hashlib
import logging
import uuid
from pathlib import Path

import json

import lmdb

# the paths of embedding_probe.common, repeated because common needs polars, which the DB venv lacks
SSD_DATA = Path('/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data')
WORK_DIR = SSD_DATA / 'embedding_probe'
PUBLIC_TEXT_LMDB = SSD_DATA / '2026-09-16.db_text_dump' / 'lmdb'

logger = logging.getLogger('KEYWORD SAMPLE')
DB = 'postgresql://librarymetadata_all:@localhost:5888/librarymetadata_all'
PAGE_TYPES = ('NormalPage', 'TitlePage', 'Advertisement', 'Table', 'TableOfContents', 'Index', 'FrontCover',
              'BackCover', 'Jacket', 'SheetMusic', 'Illustration', 'Map', 'FrontJacket', 'ListOfIllustrations',
              'Cover', 'Bibliography', 'CalibrationTable', 'ListOfTables', 'Preface', 'introduction', 'Impressum',
              'Dedication', 'ListOfMaps', 'Errata', 'Appendix', 'colophon', 'Frontispiece', 'review', 'Abstract',
              'Imprimatur')
LANGUAGES = ('ces', 'deu', 'eng', 'fra', 'lat', 'pol', 'hun', 'rus', 'ita', 'spa', 'dan', 'slk', 'ukr', 'nor', 'swe')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--candidates', type=int, default=40000, help='Candidate pages per page type.')
    parser.add_argument('--per-cell', type=int, default=60, help='Pages per (page type, language).')
    parser.add_argument('--per-doc', type=int, default=2)
    parser.add_argument('--write-used-docs', action='store_true',
                        help='Only write dev/used_docs.txt from pages.parquet and the dev sample (needs polars).')
    return parser.parse_args(argv)


def candidates(cur, page_type: str, n: int):
    cur.execute("select count(*) from meta_records where record_type = 'page' and page_type = %s", (page_type,))
    total = cur.fetchone()[0]
    if total <= n:
        cur.execute("select id::text, library, parent_id::text from meta_records "
                    "where record_type = 'page' and page_type = %s", (page_type,))
    else:
        percent = min(100.0, 100.0 * n / total * 3)
        cur.execute(f"select id::text, library, parent_id::text from meta_records tablesample system ({percent}) "
                    "where record_type = 'page' and page_type = %s limit %s", (page_type, n))
    return cur.fetchall(), total


def ancestors(cur, parent_ids):
    """parent id -> (languages, top-level record type, top-level id), walking up the parent chain."""
    info = {p: [None, None, p] for p in parent_ids if p}
    current = {p: p for p in info}  # start parent -> current ancestor
    for _ in range(6):
        ids = sorted({a for a in current.values() if a})
        if not ids:
            break
        cur.execute("select id::text, parent_id::text, record_type, metadata_json::jsonb -> 'Language' "
                    "from meta_records where id = any(%s::uuid[])", (ids,))
        rows = {r[0]: r[1:] for r in cur.fetchall()}
        nxt = {}
        for start, a in current.items():
            if a not in rows:
                continue
            parent, record_type, language = rows[a]
            if language and info[start][0] is None:
                info[start][0] = sorted({lang for group in language for lang in group}
                                        if isinstance(language[0], list) else set(language))
            info[start][1], info[start][2] = record_type, a
            if parent:
                nxt[start] = parent
        current = nxt
    return info


def main(argv=None):
    args = parse_arguments(argv)
    used_path = args.work_dir / 'dev' / 'used_docs.txt'
    if args.write_used_docs:
        import polars as pl
        used = set(pl.read_parquet(args.work_dir / 'pages.parquet')['doc'].to_list())
        used |= set(pl.read_parquet(args.work_dir / 'dev' / 'dev_pages.parquet')['doc'].to_list())
        used_path.write_text('\n'.join(sorted(used)) + '\n')
        return
    import psycopg2
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.work_dir / 'logs' / 'build_keyword_sample.txt')])
    used = set(used_path.read_text().split())
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    rows = []
    with psycopg2.connect(DB) as conn, conn.cursor() as cur, env.begin() as txn:
        for page_type in PAGE_TYPES:
            pages, total = candidates(cur, page_type, args.candidates * (4 if page_type == 'NormalPage' else 1))
            pages = [p for p in pages if p[2] and p[2] not in used and txn.get(f'{p[1]}_{p[0]}'.encode()) is not None]
            info = ancestors(cur, sorted({p[2] for p in pages}))
            kept, per_cell, per_doc = 0, {}, {}
            for page, library, parent in sorted(pages, key=lambda p: hashlib.blake2b(p[0].encode()).digest()):
                languages, top_type, top = info.get(parent, [None, None, parent])
                if top in used:
                    continue
                cells = [lang if lang in LANGUAGES else 'other' for lang in (languages or ['unknown'])]
                if per_doc.get(top, 0) >= args.per_doc or all(per_cell.get(c, 0) >= args.per_cell for c in cells):
                    continue
                for c in cells:
                    per_cell[c] = per_cell.get(c, 0) + 1
                per_doc[top] = per_doc.get(top, 0) + 1
                rows.append({'key': f'{library}_{page}', 'library': library, 'page': page, 'db_type': page_type,
                             'languages': languages or [], 'top_type': top_type, 'top': top})
                kept += 1
            logger.info(f'{page_type}: {total} in the DB, {len(pages)} public unused candidates, kept {kept}: '
                        + ', '.join(f'{c} {n}' for c, n in sorted(per_cell.items(), key=lambda x: -x[1])))
    out = args.work_dir / 'dev' / 'keyword_sample.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    logger.info(f'wrote {len(rows)} pages to {out}')


if __name__ == '__main__':
    main()
