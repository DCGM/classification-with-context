"""Converts the PAGE XML of an LMDB (fetch_ocr_xml.py --kind page_xml) to ALTO v2 with
embeddings/ocr_xml.py, into another LMDB with the same keys ({library}_{page uuid}) and zlib-compressed values.

For the pages on zfs1, which has PAGE XML but no ALTO zips. PAGE XML keeps no OCR logits and no word
boxes, so the words are placed along each line by their length and every word gets the confidence of
its line (PAGE XML TextEquiv conf) as WC. Lines and line boxes follow PERO OCR's own ALTO export (as on
matylda0): on 1,000 public pages the page statistics are identical to those of the earlier PERO-based
conversion. Standard library and lmdb only.

Parallel worker processes; a rerun converts only the keys not in the output yet. Logs hold counts only,
never XML content.

Example (run by the user for the zfs1 pages, see run_fetch_private_page_xml.sh):
    python -m embedding_probe.convert_page_xml_to_alto \\
        --in-lmdb  .../2026-09-16.db_text_dump.private/page_xml \\
        --out-lmdb .../2026-09-16.db_text_dump.private/alto
"""
import argparse
import logging
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import lmdb

logger = logging.getLogger('PAGE XML TO ALTO')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in-lmdb', type=Path, required=True, help='LMDB of zlib-compressed PAGE XML.')
    parser.add_argument('--out-lmdb', type=Path, required=True, help='LMDB for the zlib-compressed ALTO.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch', type=int, default=2000, help='Pages per LMDB commit.')
    parser.add_argument('--map-size-gb', type=int, default=100)
    return parser.parse_args(argv)


def convert(item: Tuple[bytes, bytes]) -> Tuple[bytes, Optional[bytes]]:
    """(key, compressed PAGE XML) -> (key, compressed ALTO, or None if the conversion failed)."""
    from embeddings.ocr_xml import read_page_xml, to_alto
    key, value = item
    try:
        page = read_page_xml(zlib.decompress(value))
        if page is None:
            return key, None
        return key, zlib.compress(to_alto(page, page_id=key.decode('utf-8').rsplit('_', 1)[1]), 6)
    except Exception:  # noqa: BLE001 -- counted, never logged with content
        return key, None


def main(argv=None):
    args = parse_arguments(argv)
    args.out_lmdb.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='PAGE XML TO ALTO - %(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_lmdb / 'convert_log.txt', encoding='utf-8')])
    logger.info(' '.join(sys.argv))
    source = lmdb.open(str(args.in_lmdb), readonly=True, lock=False, readahead=False)
    target = lmdb.open(str(args.out_lmdb), map_size=args.map_size_gb * 1024 ** 3)
    with target.begin() as txn:
        done = {key for key, _ in txn.cursor()}
    with source.begin() as txn:
        keys = [key for key, _ in txn.cursor() if key not in done]
    logger.info(f'{len(keys)} pages to convert ({len(done)} converted before)')
    converted = failed = 0
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for start in range(0, len(keys), args.batch):
            with source.begin() as txn:
                items = [(key, txn.get(key)) for key in keys[start:start + args.batch]]
            results = list(pool.map(convert, items, chunksize=50))
            with target.begin(write=True) as txn:
                for key, value in results:
                    if value is None:
                        failed += 1
                    else:
                        txn.put(key, value)
                        converted += 1
            logger.info(f'{min(start + args.batch, len(keys))} of {len(keys)}: {converted} converted, '
                        f'{failed} failed, {time.time() - started:.0f}s')
    source.close()
    target.close()
    logger.info(f'finished: {converted} converted, {failed} failed')


if __name__ == '__main__':
    main()
