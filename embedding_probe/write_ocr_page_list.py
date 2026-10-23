"""Writes the list of pages whose OCR XML the experiments need, for fetch_ocr_xml.py: the annotated pages
with their context windows (pages.parquet, up to its max context) and the DB-labelled public
development sample (dev/dev_pages.parquet, if built).

Output <work-dir>/ocr_pages.tsv: library <TAB> page uuid, one page per line, sorted, no header.
Page ids only -- no paths, no text.

Example:
    python -m embedding_probe.write_ocr_page_list
"""
import argparse
from pathlib import Path

import polars as pl

from embedding_probe.common import WORK_DIR


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    pages = pl.read_parquet(args.work_dir / 'pages.parquet')
    window = (pages.select('library', pl.col('window_page').alias('page')).explode('page')
              .filter(pl.col('page').is_not_null()))
    parts = [window]
    dev = args.work_dir / 'dev' / 'dev_pages.parquet'
    if dev.exists():
        parts.append(pl.read_parquet(dev).select('library', 'page'))
    result = pl.concat(parts).unique().sort('library', 'page')
    result.write_csv(args.work_dir / 'ocr_pages.tsv', separator='\t', include_header=False)
    print(f'{len(result)} pages ({len(window.unique())} from the annotated windows) -> {args.work_dir / "ocr_pages.tsv"}')


if __name__ == '__main__':
    main()
