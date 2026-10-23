"""Development tool: a sample of DB-labelled pages with PUBLIC text from the trn context file, to
design the text cues on more examples than the annotated pages have for the text-heavy page types.

Up to --per-class pages per DB page type (at most --per-doc per document and type), only pages whose
text is in the public text LMDB; the private text LMDB is never opened. The trn context file holds
no tst document, so tst stays untouched. Candidates are drawn by a stable hash of the page key, with
a per-type rate from the type's count, in one streaming pass over the CSV.

Writes <work-dir>/dev/dev_pages.parquet (library, doc, page, key, db_type, order, rank, doc_pages,
placement, path; rank/doc_pages are not filled) -- page ids and DB metadata only.

Example:
    python -m embedding_probe.build_dev_sample --csv-dir /home/ikohut/tmp/consolidation/out
"""
import argparse
import logging
from pathlib import Path

import lmdb
import polars as pl

from embedding_probe.common import CONTEXT_CSV, CSV_DIR, PAGE_TYPES, PUBLIC_TEXT_LMDB, WORK_DIR, scan_csv, setup_logging

logger = logging.getLogger('DEV SAMPLE')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--csv-dir', type=Path, default=CSV_DIR)
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--per-class', type=int, default=500)
    parser.add_argument('--per-doc', type=int, default=3)
    parser.add_argument('--candidates', type=int, default=20000,
                        help='Expected candidates per type before the public-text and per-document filters.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    out_dir = args.work_dir / 'dev'
    setup_logging('DEV SAMPLE', args.work_dir / 'logs' / 'build_dev_sample.txt')
    lf = scan_csv(args.csv_dir / CONTEXT_CSV['trn']).filter(pl.col('db_type').is_in(list(PAGE_TYPES)))
    counts = dict(lf.group_by('db_type').len().collect(engine='streaming').iter_rows())
    logger.info(f'DB page types in the trn context: {counts}')
    rates = pl.DataFrame({'db_type': list(counts), 'rate': [min(1.0, args.candidates / n) for n in counts.values()]})
    candidates = (lf.join(rates.lazy(), on='db_type')
                  .with_columns(pl.format('{}_{}', 'library', 'page').alias('key'))
                  .with_columns((pl.col('key').hash(seed=7) / float(2 ** 64)).alias('h'))
                  .filter(pl.col('h') < pl.col('rate'))
                  .select('library', 'doc', 'page', 'key', 'db_type', 'order', 'placement', 'path', 'h')
                  .collect(engine='streaming'))
    logger.info(f'{len(candidates)} candidates')
    candidates = (candidates.sort('h')
                  .with_columns(pl.int_range(pl.len()).over('db_type', 'library', 'doc').alias('in_doc'))
                  .filter(pl.col('in_doc') < args.per_doc))
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    keep = []
    with env.begin() as txn:
        taken = {}
        for row in candidates.iter_rows(named=True):
            if taken.get(row['db_type'], 0) >= args.per_class:
                continue
            if txn.get(row['key'].encode('utf-8')) is not None:  # presence only; the text is read by the explorer
                taken[row['db_type']] = taken.get(row['db_type'], 0) + 1
                keep.append(row)
    env.close()
    sample = pl.DataFrame(keep).drop('h', 'in_doc')
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(out_dir / 'dev_pages.parquet')
    logger.info(f'wrote {len(sample)} pages: {dict(sorted(taken.items(), key=lambda kv: kv[1]))}')


if __name__ == '__main__':
    main()
