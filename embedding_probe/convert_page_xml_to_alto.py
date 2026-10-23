"""Converts the PAGE XML of an LMDB (fetch_ocr_xml.py --kind page_xml) to ALTO v2 with PERO OCR's
PageLayout, into another LMDB with the same keys ({library}_{page uuid}) and zlib-compressed values.

For the pages on zfs1, which has PAGE XML but no ALTO zips. The result matches PERO's own ALTO export
(as on matylda0) in lines and words. PAGE XML keeps no OCR logits, so PERO cannot align the words
inside a line (its fallback gives all words of a line the same width, overrunning the line); the
words are therefore placed here by their length: each word gets the share of the line width of its
characters, a space counting as one character, with SP elements between words like PERO's own
export. Every word gets the confidence of its line (PAGE XML TextEquiv conf) as WC, since PAGE XML
has no word confidences.

Runs in the pero-ocr venv (pero_ocr, lmdb). Parallel worker processes; a rerun converts only the
keys not in the output yet. Logs hold counts only, never XML content.

Example (run by the user for the zfs1 pages, see run_fetch_private_page_xml.sh):
    /home/ikohut/python_env/pero-ocr/bin/python3 -m embedding_probe.convert_page_xml_to_alto \\
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


def line_confidences(page_xml: bytes) -> dict:
    """PAGE XML TextLine id -> its TextEquiv conf."""
    from lxml import etree
    result = {}
    for line in etree.fromstring(page_xml).iter('{*}TextLine'):
        equiv = line.find('{*}TextEquiv')
        if equiv is not None and equiv.get('conf') is not None:
            result[line.get('id')] = equiv.get('conf')
    return result


def place_words(alto: bytes, confidences: dict) -> bytes:
    """Re-places the words of every ALTO TextLine by their character counts and sets WC to the line's
    confidence; SP elements between the words, one character wide."""
    from lxml import etree
    root = etree.fromstring(alto)
    for line in root.iter('{*}TextLine'):
        strings = line.findall('{*}String')
        for sp in line.findall('{*}SP'):
            line.remove(sp)
        if not strings:
            continue
        hpos, vpos = float(line.get('HPOS', 0)), line.get('VPOS', '0')
        width, height = float(line.get('WIDTH', 0)), line.get('HEIGHT', '0')
        lengths = [max(len(s.get('CONTENT', '')), 1) for s in strings]
        unit = width / (sum(lengths) + len(lengths) - 1)
        conf = confidences.get(line.get('ID', '').removeprefix('line_'))
        x = hpos
        for i, (string, n) in enumerate(zip(strings, lengths)):
            string.set('HPOS', str(int(round(x))))
            string.set('WIDTH', str(int(round(n * unit))))
            string.set('VPOS', vpos)
            string.set('HEIGHT', height)
            if conf is not None:
                string.set('WC', conf)
            x += n * unit
            if i < len(strings) - 1:
                sp = etree.Element(string.tag[:-len('String')] + 'SP')
                sp.set('WIDTH', str(int(round(unit))))
                sp.set('VPOS', vpos)
                sp.set('HPOS', str(int(round(x))))
                string.addnext(sp)
                x += unit
    return etree.tostring(root, pretty_print=True, encoding='utf-8', xml_declaration=True)


def convert(item: Tuple[bytes, bytes]) -> Tuple[bytes, Optional[bytes]]:
    """(key, compressed PAGE XML) -> (key, compressed ALTO, or None if the conversion failed)."""
    from pero_ocr.core.layout import PageLayout
    key, value = item
    page_uuid = key.decode('utf-8').rsplit('_', 1)[1]
    try:
        page_xml = zlib.decompress(value)
        layout = PageLayout()
        layout.from_pagexml_string(page_xml.decode('utf-8'))
        alto = layout.to_altoxml_string(page_uuid=page_uuid)
        return key, zlib.compress(place_words(alto.encode('utf-8'), line_confidences(page_xml)), 6)
    except Exception:  # noqa: BLE001 -- counted, never logged with content
        return key, None


def quiet_worker():
    # PERO warns about every line it cannot align without logits; those messages are expected here
    logging.getLogger('pero_ocr').setLevel(logging.ERROR)


def main(argv=None):
    args = parse_arguments(argv)
    args.out_lmdb.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='PAGE XML TO ALTO - %(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out_lmdb / 'convert_log.txt', encoding='utf-8')])
    logger.info(' '.join(sys.argv))
    quiet_worker()
    source = lmdb.open(str(args.in_lmdb), readonly=True, lock=False, readahead=False)
    target = lmdb.open(str(args.out_lmdb), map_size=args.map_size_gb * 1024 ** 3)
    with target.begin() as txn:
        done = {key for key, _ in txn.cursor()}
    with source.begin() as txn:
        keys = [key for key, _ in txn.cursor() if key not in done]
    logger.info(f'{len(keys)} pages to convert ({len(done)} converted before)')
    converted = failed = 0
    started = time.time()
    with ProcessPoolExecutor(args.workers, initializer=quiet_worker) as pool:
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
