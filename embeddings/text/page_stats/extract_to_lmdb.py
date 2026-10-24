#!/usr/bin/env python3
"""Extracts the page statistics (page_stats.py) of all pages of the library DB into an LMDB, one sparse
vector per page, streaming the OCR zips of the documents (nothing else is stored).

Modelled on semAnt's librarymetadata/db/getters/extract_page_text_to_lmdb.py, and using its mapping file
(library, page id, zip path; sorted, so that the pages of one zip are consecutive), split by storage with
--path-prefix and resumable from the progress file. Per document zip, in a worker process:

  - <doc>.alto_xml.zip if it exists: the page's ALTO (<page id>.xml);
  - otherwise <doc>.page_xml.zip: the page's PAGE XML, converted to ALTO in memory with PERO, the words
    placed along each line by their length and given the line's confidence (as
    embedding_probe/convert_page_xml_to_alto.py does);

then embed_alto() and to_sparse(). LMDB: key '{library}_{page id}', value the sparse vector, plus the
key '__metadata__' (JSON: version, feature names, keyword groups, storage format). Progress lines and
logs hold only page ids, zip paths, statuses and exception type names, never text or XML content.

Runs in the pero-ocr venv (pero_ocr, lxml, lmdb, numpy), from the repository root.

Run with run_extract_page_stats_matylda0.sh and run_extract_page_stats_zfs1.sh (output next to the text
dumps: <data>/2026-09-16.db_page_stats_dump[.private]/{lmdb,progress.tsv,log.txt}).
"""
import argparse
import io
import itertools
import json
import logging
import multiprocessing
import os
import sys
import time
import zipfile
from collections import Counter, namedtuple
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import lmdb

from embeddings.text.page_stats.page_stats import PageStats, embed_alto, to_sparse

logger = logging.getLogger('PAGE STATS LMDB')

WorkUnit = namedtuple('WorkUnit', ['zip_path', 'pages'])
PageResult = namedtuple('PageResult', ['library', 'page_id', 'zip_path', 'status', 'reason', 'value'])
METADATA_KEY = b'__metadata__'


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--mapping-file', type=Path, required=True,
                        help='TSV of library, page id, PAGE XML zip path, sorted by path (the text dump\'s).')
    parser.add_argument('--path-prefix', type=str, default=None,
                        help='Only the zips under this prefix, e.g. /mnt/matylda0 or /mnt/zfs1.')
    parser.add_argument('--path-map', type=str, default=None, metavar='OLD=NEW',
                        help='Read the zips under NEW instead of OLD (a storage mounted elsewhere).')
    parser.add_argument('--lmdb-dir', type=Path, required=True)
    parser.add_argument('--progress-file', type=Path, default=None, help='Default: <lmdb-dir>/progress.tsv.')
    parser.add_argument('--log-file', type=Path, default=None, help='Default: <lmdb-dir>/log.txt.')
    parser.add_argument('--retry-statuses', type=str, default='',
                        help='Comma-separated statuses to retry instead of skip (missing, failed).')
    parser.add_argument('--workers', type=int, default=-1, help='Default: cpu_count().')
    parser.add_argument('--commit-every', type=int, default=20000, help='Pages per LMDB commit.')
    parser.add_argument('--map-size-gb', type=int, default=60)
    parser.add_argument('--max-zips', type=int, default=None, help='Stop after this many zips (for a trial run).')
    parser.add_argument('--logging-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args(argv)


def alto_zip_path(page_xml_zip: str) -> str:
    return page_xml_zip[:-len('.page_xml.zip')] + '.alto_xml.zip' if page_xml_zip.endswith('.page_xml.zip') \
        else page_xml_zip


def read_mapping_groups(mapping_path: Path, path_prefix: Optional[str]) -> Iterator[Tuple[str, List[Tuple[str, str]]]]:
    def rows():
        with mapping_path.open('r', encoding='utf-8') as f:
            for line in f:
                library, page_id, zip_path = line.rstrip('\n').split('\t')
                yield library, page_id, zip_path
    for zip_path, group in itertools.groupby(rows(), key=lambda r: r[2]):
        if path_prefix and not zip_path.startswith(path_prefix):
            continue
        yield zip_path, [(library, page_id) for library, page_id, _ in group]


def load_progress(progress_path: Path, retry: Set[str]) -> Set[str]:
    recorded = set()
    if progress_path.exists():
        with progress_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 4 and parts[3] not in retry:
                    recorded.add(f'{parts[0]}_{parts[1]}')
    return recorded


def page_xml_to_alto(page_xml: bytes, page_id: str) -> bytes:
    """PAGE XML -> ALTO with PERO, words placed by length (embedding_probe/convert_page_xml_to_alto.py)."""
    from pero_ocr.core.layout import PageLayout
    from embedding_probe.convert_page_xml_to_alto import line_confidences, place_words
    layout = PageLayout()
    layout.from_pagexml_string(page_xml.decode('utf-8'))
    alto = layout.to_altoxml_string(page_uuid=page_id)
    return place_words(alto.encode('utf-8'), line_confidences(page_xml))


def open_zip(path: str) -> Optional[zipfile.ZipFile]:
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return zipfile.ZipFile(io.BytesIO(f.read()))


def process_zip(unit: WorkUnit, path_map: Optional[Tuple[str, str]] = None) -> List[PageResult]:
    zip_path, pages = unit
    local = zip_path.replace(path_map[0], path_map[1], 1) if path_map else zip_path
    kind = 'alto'
    try:
        archive = open_zip(alto_zip_path(local))
        if archive is None:
            kind, archive = 'page_xml', open_zip(local)
    except (OSError, zipfile.BadZipFile) as exc:
        return [PageResult(lib, page, zip_path, 'missing', f'archive_error:{type(exc).__name__}', None)
                for lib, page in pages]
    if archive is None:
        return [PageResult(lib, page, zip_path, 'missing', 'archive_missing', None) for lib, page in pages]
    names = set(archive.namelist())
    results = []
    for library, page_id in pages:
        member = f'{page_id}.xml'
        if member not in names:
            results.append(PageResult(library, page_id, zip_path, 'missing', f'{kind}_member_missing', None))
            continue
        try:
            xml = archive.read(member)
            if kind == 'page_xml':
                xml = page_xml_to_alto(xml, page_id)
            value = to_sparse(embed_alto(xml))
        except Exception as exc:  # noqa: BLE001 -- the type only, never the message (it may quote content)
            results.append(PageResult(library, page_id, zip_path, 'failed', f'{kind}:{type(exc).__name__}', None))
            continue
        results.append(PageResult(library, page_id, zip_path, f'done_{kind}', '', value))
    return results


def quiet_worker():
    # PERO warns about every line it cannot align without logits; expected for PAGE XML without them
    logging.getLogger('pero_ocr').setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)


def run(args) -> None:
    progress_path = args.progress_file or args.lmdb_dir / 'progress.tsv'
    args.lmdb_dir.mkdir(parents=True, exist_ok=True)
    retry = {s.strip() for s in args.retry_statuses.split(',') if s.strip()}
    recorded = load_progress(progress_path, retry)
    path_map = tuple(args.path_map.split('=', 1)) if args.path_map else None

    def units() -> Iterator[WorkUnit]:
        for n, (zip_path, pages) in enumerate(read_mapping_groups(args.mapping_file, args.path_prefix)):
            if args.max_zips is not None and n >= args.max_zips:
                return
            pending = [(lib, page) for lib, page in pages if f'{lib}_{page}' not in recorded]
            if pending:
                yield WorkUnit(zip_path, pending)

    env = lmdb.open(str(args.lmdb_dir), map_size=args.map_size_gb * 1024 ** 3)
    with env.begin(write=True) as txn:
        txn.put(METADATA_KEY, json.dumps(PageStats().metadata()).encode('utf-8'))
    workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    logger.info(f'path prefix {args.path_prefix!r}, {len(recorded)} pages recorded before, {workers} workers')
    counts: Counter = Counter()
    started = time.time()

    def flush(buffer: List[PageResult]) -> None:
        if not buffer:
            return
        with env.begin(write=True) as txn:
            for r in buffer:
                if r.value is not None:
                    txn.put(f'{r.library}_{r.page_id}'.encode('utf-8'), r.value)
        with progress_path.open('a', encoding='utf-8') as f:
            for r in buffer:
                f.write(f'{r.library}\t{r.page_id}\t{r.zip_path}\t{r.status}\t{r.reason}\n')
            f.flush()
            os.fsync(f.fileno())
        counts.update(r.status for r in buffer)
        rate = sum(counts.values()) / max(time.time() - started, 1e-9)
        logger.info(' '.join(f'{k}={v}' for k, v in sorted(counts.items())) + f' | {rate:.0f} pages/s')

    import functools
    try:
        buffer: List[PageResult] = []
        with multiprocessing.Pool(workers, initializer=quiet_worker) as pool:
            for results in pool.imap_unordered(functools.partial(process_zip, path_map=path_map), units(),
                                               chunksize=1):
                buffer.extend(results)
                if len(buffer) >= args.commit_every:
                    flush(buffer)
                    buffer = []
        flush(buffer)
        env.sync()
    finally:
        env.close()
    logger.info('finished: ' + ' '.join(f'{k}={v}' for k, v in sorted(counts.items())))


def main(argv=None):
    args = parse_arguments(argv)
    args.lmdb_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=args.logging_level, format='PAGE STATS LMDB - %(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file or args.lmdb_dir / 'log.txt', encoding='utf-8')])
    logger.info(' '.join(sys.argv))
    run(args)


if __name__ == '__main__':
    main()
