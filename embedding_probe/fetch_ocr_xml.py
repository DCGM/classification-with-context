"""Fetches the OCR XML (PAGE XML by default, or ALTO) of given pages from the document zips into an LMDB.

The page list (library <TAB> page uuid, from write_ocr_page_list.py) is looked up in the library
metadata DB (meta_records.image_path, via the tunnel on localhost:5888). The zip of a page is derived
from its image path like the PAGE-XML zip of the text dump (the DB path columns may be stale):

    .../<docdir>.images/<page>.jpg  ->  .../<docdir>.<kind>.zip  (member <page>.xml)

with <kind> page_xml (default) or alto_xml. Both storages have the PAGE XML (the text dumps were
extracted from it); only matylda0 also has ALTO, so the experiments use PAGE XML everywhere.
<docdir> is the document or a higher level (e.g. a whole periodical), so one zip serves many pages.
--path-map OLD=NEW rewrites the start of the DB paths (for storage mounted elsewhere), and only zips
under --path-prefix are opened; pages outside it are counted and skipped. Zips are read in parallel
threads, each reading only the members it needs.

Output: an LMDB (--out-lmdb, e.g. <text dump>/page_xml next to <text dump>/lmdb) with key
'{library}_{page uuid}' (the text LMDB key) and value the zlib-compressed XML, plus
<out-lmdb>/progress.tsv (library, page, zip path, status: done | missing_zip | missing_member |
bad_zip | no_image_path). Pages outside --path-prefix are only counted (outside_prefix), not
recorded, so a run for another storage can fetch them. A rerun skips pages already done.

Only the standard library, lmdb and psycopg2 are needed (e.g. the db_text_dump venv). Logs hold
counts and paths only, never XML content.

matylda0 pages (the run for the experiments):
    /home/ikohut/python_env/db_text_dump/bin/python3 -m embedding_probe.fetch_ocr_xml \\
        --pages /run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/embedding_probe/ocr_pages.tsv \\
        --path-prefix /mnt/matylda0 \\
        --out-lmdb /run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump/page_xml

zfs1 pages: run by the user, see run_fetch_private_page_xml.sh.
"""
import argparse
import logging
import os
import re
import sys
import threading
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb

logger = logging.getLogger('FETCH OCR XML')
UUID = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
DEFAULT_DB_URL = 'postgresql://librarymetadata_all:@localhost:5888/librarymetadata_all'


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--pages', type=Path, required=True, help='TSV: library <TAB> page uuid.')
    parser.add_argument('--out-lmdb', type=Path, required=True, help='Output LMDB directory.')
    parser.add_argument('--kind', choices=('page_xml', 'alto_xml'), default='page_xml',
                        help='Which zips to read: <docdir>.page_xml.zip or <docdir>.alto_xml.zip.')
    parser.add_argument('--path-prefix', required=True,
                        help='Only zips whose (mapped) path starts with this are opened.')
    parser.add_argument('--path-map', action='append', default=[],
                        help='OLD=NEW: rewrite DB paths starting with OLD to start with NEW (repeatable).')
    parser.add_argument('--db-url', default=os.environ.get('DATABASE_URL', DEFAULT_DB_URL))
    parser.add_argument('--dry-run', action='store_true',
                        help='Only look the pages up in the DB and print the zip path prefixes (first 4 directories)'
                             ' with page counts, to choose --path-map/--path-prefix; opens no file '
                             'and writes nothing.')
    parser.add_argument('--diagnose', type=int, default=0,
                        help='For the first N documents whose zip is missing, log the names of the files '
                             'next to it (<docdir>.*), to see how the storage names things; names only.')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--map-size-gb', type=int, default=100)
    parser.add_argument('--commit-every', type=int, default=2000, help='Pages per LMDB commit.')
    return parser.parse_args(argv)


def ocr_zip_path(image_path: str, kind: str = 'page_xml') -> str:
    """.../<docdir>.images/<page>.jpg -> .../<docdir>.<kind>.zip"""
    parent = os.path.dirname(image_path)
    if parent.endswith('.images'):
        parent = parent[:-len('.images')]
    return f'{parent}.{kind}.zip'


def map_path(path: str, path_map: List[Tuple[str, str]]) -> str:
    for old, new in path_map:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def read_page_list(path: Path) -> List[Tuple[str, str]]:
    with path.open(encoding='utf-8') as f:
        return [tuple(line.rstrip('\n').split('\t')[:2]) for line in f if line.strip()]


def image_paths(db_url: str, pages: List[Tuple[str, str]], batch: int = 5000) -> Dict[Tuple[str, str], str]:
    """(library, page) -> image_path from meta_records."""
    import psycopg2
    result = {}
    valid = [p for p in pages if UUID.fullmatch(p[1])]
    if len(valid) < len(pages):
        logger.info(f'{len(pages) - len(valid)} page ids are not UUIDs and cannot be looked up')
    with psycopg2.connect(db_url) as connection, connection.cursor() as cursor:
        for start in range(0, len(valid), batch):
            chunk = valid[start:start + batch]
            cursor.execute('SELECT library, id::text, image_path FROM meta_records WHERE id = ANY(%s::uuid[])',
                           ([p for _, p in chunk],))
            wanted = {(lib, page.lower()): (lib, page) for lib, page in chunk}
            for library, page, path in cursor.fetchall():
                key = wanted.get((library, page.lower()))
                if key is not None and path and path != 'not_found':
                    result[key] = path
            logger.info(f'DB: {min(start + batch, len(valid))} of {len(valid)} pages looked up, '
                        f'{len(result)} with an image path')
    return result


def read_members(zip_path: str, pages: List[Tuple[str, str]]):
    """[(library, page, status, compressed XML or None)] for the pages of one zip."""
    if not os.path.exists(zip_path):
        return [(lib, page, 'missing_zip', None) for lib, page in pages]
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            out = []
            for lib, page in pages:
                member = f'{page}.xml'
                if member in names:
                    out.append((lib, page, 'done', zlib.compress(zf.read(member), 6)))
                else:
                    out.append((lib, page, 'missing_member', None))
            return out
    except (OSError, zipfile.BadZipFile):
        return [(lib, page, 'bad_zip', None) for lib, page in pages]


def main(argv=None):
    args = parse_arguments(argv)
    args.out_lmdb.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='FETCH OCR XML - %(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_lmdb / 'log.txt', encoding='utf-8')])
    logger.info(' '.join(sys.argv))
    path_map = [tuple(m.split('=', 1)) for m in args.path_map]
    progress_path = args.out_lmdb / 'progress.tsv'
    done = set()
    if progress_path.exists():
        with progress_path.open(encoding='utf-8') as f:
            done = {tuple(line.split('\t')[:2]) for line in f if line.rstrip('\n').split('\t')[3:4] == ['done']}
    pages = [p for p in read_page_list(args.pages) if p not in done]
    logger.info(f'{len(pages)} pages to fetch ({len(done)} done before)')

    paths = image_paths(args.db_url, pages)
    if args.dry_run:
        prefixes = Counter('/'.join(ocr_zip_path(p, args.kind).split('/')[:5]) for p in paths.values())
        logger.info(f'{len(paths)} of {len(pages)} pages have an image path; zip path prefixes:')
        for prefix, n in prefixes.most_common():
            mapped = map_path(prefix, path_map)
            logger.info(f'  {n:8d}  {prefix}' + (f'  -> {mapped}' if mapped != prefix else '')
                        + ('  [under --path-prefix]' if mapped.startswith(args.path_prefix) else ''))
        return
    by_zip: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    status = Counter()
    with progress_path.open('a', encoding='utf-8') as progress:
        for lib, page in pages:
            path = paths.get((lib, page))
            if path is None:
                status['no_image_path'] += 1
                progress.write(f'{lib}\t{page}\t\tno_image_path\n')
                continue
            zip_path = map_path(ocr_zip_path(path, args.kind), path_map)
            if not zip_path.startswith(args.path_prefix):
                status['outside_prefix'] += 1  # not recorded: another run (another prefix) may fetch it
                continue
            by_zip[zip_path].append((lib, page))
    logger.info(f'{sum(len(v) for v in by_zip.values())} pages in {len(by_zip)} zips under {args.path_prefix}; '
                f'skipped {dict(status)}')

    env = lmdb.open(str(args.out_lmdb), map_size=args.map_size_gb * 1024 ** 3)
    lock = threading.Lock()
    pending, written, started = [], 0, time.time()

    def flush():
        nonlocal pending, written
        with env.begin(write=True) as txn:
            for lib, page, _, value in pending:
                if value is not None:
                    txn.put(f'{lib}_{page}'.encode('utf-8'), value)
        with progress_path.open('a', encoding='utf-8') as progress:
            for lib, page, st, _ in pending:
                progress.write(f'{lib}\t{page}\t{zip_of[(lib, page)]}\t{st}\n')
        written += len(pending)
        pending = []

    zip_of = {p: z for z, ps in by_zip.items() for p in ps}
    with ThreadPoolExecutor(args.threads) as pool:
        futures = [pool.submit(read_members, z, ps) for z, ps in by_zip.items()]
        for n, future in enumerate(as_completed(futures), 1):
            results = future.result()
            with lock:
                pending.extend(results)
                status.update(r[2] for r in results)
                if len(pending) >= args.commit_every:
                    flush()
                    logger.info(f'{n} of {len(futures)} zips, {written} pages written, {dict(status)}, '
                                f'{time.time() - started:.0f}s')
    if pending:
        flush()
    env.close()
    if args.diagnose:
        missing = [z for z, ps in by_zip.items() if not os.path.exists(z)][:args.diagnose]
        for zip_path in missing:
            docdir = zip_path[:-len(f'.{args.kind}.zip')]
            parent, stem = os.path.dirname(docdir), os.path.basename(docdir)
            try:
                siblings = sorted(n for n in os.listdir(parent) if n.startswith(stem))
            except OSError as exc:
                siblings = [f'<cannot list {parent}: {exc.__class__.__name__}>']
            logger.info(f'diagnose: missing {zip_path}; next to it: {siblings or "nothing named " + stem + "*"}')
    logger.info(f'finished: {dict(status)}, {time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
