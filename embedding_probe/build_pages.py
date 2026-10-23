"""Builds the page table of the experiments: every annotated trn and tst page with its position in
its document and its context window, the pages up to --max-context places before and after it in
the same document (from the context CSV files).

Output <work-dir>/pages.parquet, one row per annotated page, in annotation order (trn, then tst):

  split, annotation_index       trn | tst, row index in the annotated file
  key                           {library}_{page uuid}: the text LMDB key, unique per page
  library, doc, page, label, db_type, placement, order, access, path   from the annotated file
                                (label = annotated page type, path = image path)
  rank, doc_pages               0-based position among the document's pages (sorted by order) and their count
  window_page, window_path      lists of 2 * max_context + 1 entries, the pages at offsets
                                -max_context .. +max_context (the page itself in the middle);
                                null where the window runs past the document's first or last page

A window never crosses into another document. The context files are scanned with streaming; only
the pages of documents holding an annotated page are collected.

Example:
    python -m embedding_probe.build_pages --max-context 3
"""
import argparse
import logging
from pathlib import Path

import polars as pl

from embedding_probe.common import ANNOTATED_CSV, CONTEXT_CSV, CSV_DIR, WORK_DIR, scan_csv, setup_logging

logger = logging.getLogger('BUILD PAGES')
DOC = ['library', 'doc']


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--csv-dir', type=Path, default=CSV_DIR, help='Directory of the four CSV files.')
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--max-context', type=int, default=3,
                        help='The largest context window K (pages on each side) the experiments will use.')
    return parser.parse_args(argv)


def windows(context: pl.DataFrame, max_context: int) -> pl.DataFrame:
    """Position and context window of every page of the given context rows (library, doc, page, order, path)."""
    context = context.with_columns(pl.col('order').cast(pl.Int64, strict=False).alias('order_int'))
    context = context.sort(DOC + ['order_int', 'page'], nulls_last=True)
    offsets = range(-max_context, max_context + 1)
    # shift(-offset) brings the page `offset` places later (offset < 0: earlier) onto this row
    return context.select(
        *DOC, 'page',
        pl.int_range(pl.len()).over(DOC).alias('rank'),
        pl.len().over(DOC).alias('doc_pages'),
        pl.concat_list([pl.col('page').shift(-o).over(DOC) for o in offsets]).alias('window_page'),
        pl.concat_list([pl.col('path').shift(-o).over(DOC) for o in offsets]).alias('window_path'),
    )


def build_split(annotated_csv: Path, context_csv: Path, split: str, max_context: int) -> pl.DataFrame:
    annotated = scan_csv(annotated_csv).collect().with_row_index('annotation_index')
    docs = annotated.select(DOC).unique()
    context = (scan_csv(context_csv).join(docs.lazy(), on=DOC, how='semi')
               .select(*DOC, 'page', 'order', 'path').collect(engine='streaming'))
    logger.info(f'{split}: {len(annotated)} annotated pages in {len(docs)} documents with {len(context)} pages')
    assert context.select(*DOC, 'page').is_duplicated().sum() == 0, 'a page is listed twice in its document'
    pages = annotated.join(windows(context, max_context), on=DOC + ['page'], how='left', maintain_order='left')
    assert len(pages) == len(annotated) and pages['rank'].null_count() == 0, 'annotated page missing in context'
    return pages.select(
        pl.lit(split).alias('split'), 'annotation_index',
        pl.format('{}_{}', 'library', 'page').alias('key'),
        'library', 'doc', 'page', pl.col('annotated_type').alias('label'), 'db_type', 'placement', 'order',
        'access', 'path', 'rank', 'doc_pages', 'window_page', 'window_path')


def main(argv=None):
    args = parse_arguments(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging('BUILD PAGES', args.work_dir / 'logs' / 'build_pages.txt')
    pages = pl.concat([build_split(args.csv_dir / ANNOTATED_CSV[s], args.csv_dir / CONTEXT_CSV[s], s,
                                   args.max_context) for s in ('trn', 'tst')])
    assert pages['key'].n_unique() == len(pages), 'a page is annotated twice'
    middle = args.max_context
    assert (pages['window_page'].list.get(middle) == pages['page']).all()
    pages.write_parquet(args.work_dir / 'pages.parquet')
    for k in range(1, args.max_context + 1):
        left = pages['window_page'].list.get(middle - k).is_null().sum()
        right = pages['window_page'].list.get(middle + k).is_null().sum()
        logger.info(f'offset {k}: no page {k} before for {left} pages, no page {k} after for {right} pages')
    logger.info(f'wrote {len(pages)} pages to {args.work_dir / "pages.parquet"}')


if __name__ == '__main__':
    main()
