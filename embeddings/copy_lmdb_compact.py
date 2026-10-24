#!/usr/bin/env python3
"""Copies an embedding LMDB (e.g. from kolosus to the SSD) compactly: the entries are read in key order and
appended, so that LMDB fills its pages (two 1,536-byte float16 vectors per 4 KB page instead of one when
the source was written in random order). Keys and values are unchanged. The small files next to the
source LMDB directory (meta.json, logs, progress and file lists) are copied too.

Resumable: an interrupted copy continues after the last key in the target.

Example:
    python -m embeddings.copy_lmdb_compact \\
        /mnt/kolosus/data/digiknihovna.data/public/text/mmBERT-base \\
        /run/media/.../data/2026-09-16.db_embedding_dump/public/text/mmBERT-base
"""
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import lmdb

logger = logging.getLogger('COPY LMDB')
SIDE_FILES = ('meta.json', 'log.txt', 'console.log', 'progress.tsv', 'file_list.tsv')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('source', type=Path, help='Directory with lmdb/ (and meta.json, ...).')
    parser.add_argument('target', type=Path)
    parser.add_argument('--map-size-gb', type=int, default=150)
    parser.add_argument('--commit-every', type=int, default=200000)
    return parser.parse_args(argv)


def copy(source: Path, target: Path, map_size_gb: int, commit_every: int) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in SIDE_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    src = lmdb.open(str(source / 'lmdb'), readonly=True, lock=False, readahead=True)
    dst = lmdb.open(str(target / 'lmdb'), map_size=map_size_gb * 1024 ** 3)
    total = src.stat()['entries']
    with dst.begin() as txn:
        done = txn.stat()['entries']
        cursor = txn.cursor()
        last = bytes(cursor.key()) if done and cursor.last() else None
    logger.info(f'{source} -> {target}: {total} entries, {done} copied before')
    started, copied = time.time(), 0

    def write(batch):
        nonlocal copied
        if not batch:
            return
        with dst.begin(write=True) as d:
            for k, v in batch:
                d.put(k, v, append=True)
        copied += len(batch)
        rate = copied / (time.time() - started)
        logger.info(f'{done + copied}/{total} | {rate:.0f} entries/s | '
                    f'{(total - done - copied) / max(rate, 1) / 3600:.1f} h left')

    with src.begin() as s:
        cursor = s.cursor()
        positioned = cursor.set_range(last) if last is not None else cursor.first()
        if positioned and last is not None and cursor.key() == last:
            positioned = cursor.next()
        batch = []
        for k, v in (cursor.iternext() if positioned else ()):
            batch.append((k, v))
            if len(batch) >= commit_every:
                write(batch)
                batch = []
        write(batch)
    dst.sync()
    stat = dst.stat()
    logger.info(f'finished: {stat["entries"]} entries, {stat["leaf_pages"]} leaf pages')
    dst.close()
    src.close()


def main(argv=None):
    args = parse_arguments(argv)
    args.target.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='COPY LMDB - %(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.target / 'copy_log.txt', encoding='utf-8')])
    copy(args.source, args.target, args.map_size_gb, args.commit_every)


if __name__ == '__main__':
    main()
