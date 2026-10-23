"""Development tool: per page type, the words and line patterns typical of the PUBLIC text of the
annotated trn pages, to design keyword and layout cues (see text_cues.py).

Reads only the public text LMDB; it has no option for the private one, and must never get one (see
CLAUDE.md). Only trn pages are used (the annotated trn pages, or the DB-labelled public sample of
build_dev_sample.py from the trn context file), so tst stays untouched during development.

Per class it prints the words found on pages of the most distinct documents of that class, ranked
by a document-level log-odds against the other classes (a word counts once per document and class,
so one long book cannot dominate), and the share of the class's pages that contain a word.

Example:
    python -m embedding_probe.explore_public_text --top 25
    python -m embedding_probe.explore_public_text --classes Impressum Colophon --show 3
"""
import argparse
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import polars as pl

from embedding_probe.common import PUBLIC_TEXT_LMDB, WORK_DIR

WORD = re.compile(r'[^\W\d_]{3,}')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--source', choices=('dev', 'annotated'), default='dev',
                        help='dev: the DB-labelled public sample of build_dev_sample.py (label = DB page type); '
                             'annotated: the annotated trn pages with public text.')
    parser.add_argument('--classes', nargs='*', default=None, help='Only these classes (default: all).')
    parser.add_argument('--top', type=int, default=20, help='Words per class.')
    parser.add_argument('--min-docs', type=int, default=3, help='A word must occur in this many documents of the class.')
    parser.add_argument('--region', choices=('page', 'head'), default='page',
                        help='Words of the whole page, or of its first 3 non-empty lines (headings).')
    parser.add_argument('--rank', choices=('lift', 'log-odds'), default='lift',
                        help='lift: share of the class documents / share of all documents, for words on at '
                             'least --min-share of the class documents; log-odds favours rarer, more specific words.')
    parser.add_argument('--min-share', type=float, default=0.04)
    parser.add_argument('--show', type=int, default=0, help='Also print the first lines of this many pages per class.')
    return parser.parse_args(argv)


def normalize(text: str) -> str:
    """Lower case without diacritics, so OCR and spelling variants of a word meet."""
    text = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in text if not unicodedata.combining(c))


def words(text: str) -> set:
    return set(WORD.findall(normalize(text)))


def load_pages(work_dir: Path, source: str) -> pl.DataFrame:
    """key, label, doc of the development pages (trn only)."""
    if source == 'dev':
        return pl.read_parquet(work_dir / 'dev' / 'dev_pages.parquet').select(
            'key', pl.col('db_type').alias('label'), pl.format('{}/{}', 'library', 'doc').alias('doc'))
    return pl.read_parquet(work_dir / 'pages.parquet').filter(pl.col('split') == 'trn').select('key', 'label', 'doc')


def main(argv=None):
    args = parse_arguments(argv)
    pages = load_pages(args.work_dir, args.source)
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    rows = []
    with env.begin() as txn:
        for key, label, doc in pages.select('key', 'label', 'doc').iter_rows():
            value = txn.get(key.encode('utf-8'))
            if value is not None:
                rows.append((label, doc, value.decode('utf-8')))
    env.close()
    print(f'{len(rows)} trn pages with public text')

    # document-level counts: word -> number of (class, document) pairs it occurs in
    docs_of = defaultdict(set)
    pages_of = Counter()
    word_docs = defaultdict(lambda: defaultdict(set))
    word_pages = defaultdict(Counter)
    for label, doc, text in rows:
        docs_of[label].add(doc)
        pages_of[label] += 1
        region = text if args.region == 'page' else '\n'.join(
            [line for line in text.split('\n') if line.strip()][:3])
        for w in words(region):
            word_docs[label][w].add(doc)
            word_pages[label][w] += 1
    total_docs = sum(len(d) for d in docs_of.values())
    all_word_docs = Counter()
    for label in word_docs:
        for w, d in word_docs[label].items():
            all_word_docs[w] += len(d)

    classes = args.classes or sorted(docs_of, key=lambda c: -pages_of[c])
    for label in classes:
        n_docs, n_pages = len(docs_of[label]), pages_of[label]
        other_docs = total_docs - n_docs
        scored = []
        for w, d in word_docs[label].items():
            if len(d) < args.min_docs:
                continue
            inside = len(d)
            outside = all_word_docs[w] - inside
            if args.rank == 'lift':
                if inside / n_docs < args.min_share:
                    continue
                score = (inside / n_docs) / (all_word_docs[w] / total_docs)
            else:
                score = (math.log((inside + 0.5) / (n_docs - inside + 0.5))
                         - math.log((outside + 0.5) / (other_docs - outside + 0.5)))
            scored.append((score, w, inside, outside, word_pages[label][w] / n_pages))
        scored.sort(reverse=True)
        print(f'\n== {label}: {n_pages} pages, {n_docs} documents')
        print('   ' + ', '.join(f'{w} ({inside}d/{outside}o, {share:.0%})'
                                 for _, w, inside, outside, share in scored[:args.top]))
        for _, _, text in [r for r in rows if r[0] == label][:args.show]:
            print('   ---\n   ' + '\n   '.join(text.strip().split('\n')[:12]))


if __name__ == '__main__':
    main()
