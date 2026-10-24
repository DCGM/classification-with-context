"""Development tool: headings of the front and back matter of PUBLIC monographs per language, to find
missing keyword stems of the page statistics in the less frequent languages.

From the library metadata DB: per language up to --docs top-level monographs in that language that are
not used so far (dev/used_docs.txt) and whose pages are in the PUBLIC text LMDB (the private one is never
opened); of each, the first and last --edge pages by order. Per language: the most frequent heading words
(first 3 lines, folded like the page statistics) that no stem covers, by number of documents, and short
heading lines in which no stem matches. Runs in the db_text_dump venv (psycopg2, lmdb); stdlib otherwise.

Writes <work-dir>/dev/front_back_headings.md.

Example:
    python -m embedding_probe.front_back_headings
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import lmdb

SSD_DATA = Path('/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data')
WORK_DIR = SSD_DATA / 'embedding_probe'
PUBLIC_TEXT_LMDB = SSD_DATA / '2026-09-16.db_text_dump' / 'lmdb'
DB = 'postgresql://librarymetadata_all:@localhost:5888/librarymetadata_all'
LANGUAGES = ('deu', 'eng', 'fra', 'lat', 'pol', 'hun', 'rus', 'ita', 'spa', 'dan', 'slk', 'ukr', 'nor', 'swe', 'ces')
WORD = re.compile(r'[^\W\d_]{3,}')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--docs', type=int, default=150, help='Documents per language.')
    parser.add_argument('--edge', type=int, default=8, help='Pages from each end of a document.')
    parser.add_argument('--top', type=int, default=60)
    parser.add_argument('--lines', type=int, default=40)
    return parser.parse_args(argv)


def fold(text):
    text = unicodedata.normalize('NFKD', text.lower().replace('ß', 'ss'))
    return ''.join(c for c in text if not unicodedata.combining(c))


def main(argv=None):
    args = parse_arguments(argv)
    import psycopg2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from embeddings.text.page_stats.page_stats import STEMS, normalize  # numpy only
    stems = [re.compile(r'(?<![^\W\d_])' + re.escape(s)) for s in STEMS]
    used = set((args.work_dir / 'dev' / 'used_docs.txt').read_text().split())
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    out = ['# Front and back matter headings of public monographs per language', '',
           'Heading = first 3 lines. Words: "word (documents)", not covered by a stem. Lines: short headings '
           '(at most 6 words) in which no stem matches.', '']
    with psycopg2.connect(DB) as conn, conn.cursor() as cur, env.begin() as txn:
        for lang in LANGUAGES:
            cur.execute("select id::text, library from meta_records tablesample system (5) where record_type = 'monograph' "
                        "and metadata_json::jsonb -> 'Language' @> %s::jsonb limit %s",
                        (f'[["{lang}"]]', args.docs * 6))
            docs = [d for d in cur.fetchall() if d[0] not in used]
            words, lines, n_docs, n_pages = Counter(), Counter(), 0, 0
            for doc, library in docs:
                if n_docs >= args.docs:
                    break
                cur.execute('select id::text from meta_records where parent_id = %s::uuid and record_type = \'page\' '
                            'order by "order"', (doc,))
                pages = [p[0] for p in cur.fetchall()]
                edge = pages[:args.edge] + pages[-args.edge:] if len(pages) > 2 * args.edge else pages
                texts = [txn.get(f'{library}_{p}'.encode()) for p in edge]
                texts = [t.decode('utf-8') for t in texts if t is not None]
                if not texts:
                    continue
                n_docs += 1
                n_pages += len(texts)
                doc_words = set()
                for text in texts:
                    head = [line.strip() for line in text.split('\n') if line.strip()][:3]
                    norm = ' ' + normalize('\n'.join(head)).replace('\n', ' ') + ' '
                    doc_words |= {w for w in WORD.findall(norm)
                                  if not any(s.match(w) for s in stems if ' ' not in s.pattern)}
                    for line in head:
                        n = ' ' + normalize(line) + ' '
                        if 1 <= len(line.split()) <= 6 and sum(c.isalpha() for c in line) >= 4 \
                                and not any(s.search(n) for s in stems):
                            lines[line[:80]] += 1
                words.update(doc_words)
            out += [f'## {lang}: {n_docs} documents, {n_pages} pages', '',
                    ', '.join(f'{w} ({n})' for w, n in words.most_common(args.top) if n >= 2) or '-', '']
            out += [f'- `{line}` ({n})' for line, n in lines.most_common(args.lines)] + ['']
            print(lang, n_docs, n_pages, flush=True)
    path = args.work_dir / 'dev' / 'front_back_headings.md'
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
