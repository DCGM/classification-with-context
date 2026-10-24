"""Development tool: candidate keyword stems from the varied PUBLIC sample of build_keyword_sample.py.

Per DB page type (overall and per document language), the words (folded like the page statistics:
lower case, no diacritics) that are in more of that type's pages than of the other sampled pages, in
the whole page and in the first 3 lines (headings), and that no stem of embeddings/text/page_stats
covers yet; plus a few headings per type and language. Only public text is read.

Writes <work-dir>/dev/keyword_candidates.md.

Example:
    python -m embedding_probe.keyword_candidates
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import lmdb

from embedding_probe.common import PUBLIC_TEXT_LMDB, WORK_DIR
from embeddings.text.page_stats.page_stats import EDGE_LINES, STEMS, normalize

WORD = re.compile(r'[^\W\d_]{3,}')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--top', type=int, default=25)
    parser.add_argument('--min-docs', type=int, default=3)
    parser.add_argument('--headings', type=int, default=6)
    return parser.parse_args(argv)


def covered(word: str) -> bool:
    return any(word.startswith(s.strip()) for s in STEMS if ' ' not in s.strip())


def main(argv=None):
    args = parse_arguments(argv)
    rows = [json.loads(line) for line in open(args.work_dir / 'dev' / 'keyword_sample.jsonl', encoding='utf-8')]
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    pages = []
    with env.begin() as txn:
        for r in rows:
            text = txn.get(r['key'].encode()).decode('utf-8')
            lines = [line.strip() for line in normalize(text).split('\n') if line.strip()]
            raw = [line.strip() for line in text.split('\n') if line.strip()]
            pages.append((r, set(WORD.findall(' '.join(lines))), set(WORD.findall(' '.join(lines[:EDGE_LINES]))),
                          ' | '.join(raw[:EDGE_LINES])[:160]))
    cache = {}

    def is_covered(w):
        if w not in cache:
            cache[w] = covered(w)
        return cache[w]

    def table(target, background, top_docs):
        n_t, n_b = len(target), len(background)
        df_t = Counter(w for words in target for w in words)
        df_b = Counter(w for words in background for w in words)
        scored = []
        for w, a in df_t.items():
            if top_docs[w] < args.min_docs or is_covered(w):
                continue
            score = math.log((a + 0.5) / (n_t + 1)) - math.log((df_b[w] + 0.5) / (n_b + 1))
            if score > 1.0:
                scored.append((score * math.sqrt(a), w, a, df_b[w]))
        scored.sort(reverse=True)
        return ', '.join(f'{w} ({a}/{n_t} vs {b}/{n_b})' for _, w, a, b in scored[:args.top]) or '-'

    by_type = defaultdict(list)
    for p in pages:
        by_type[p[0]['db_type']].append(p)
    out = ['# Keyword candidates from the varied public sample', '',
           f'{len(pages)} pages. Per page type: words in more of its pages than of the other sampled pages, '
           'not covered by a stem; "a/n vs b/m": pages of the type / other pages with the word. '
           f'Words in at least {args.min_docs} documents.', '']
    for page_type, group in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if page_type == 'NormalPage':
            continue
        others = [p for p in pages if p[0]['db_type'] != page_type]
        out += [f'## {page_type} ({len(group)} pages)', '']
        languages = Counter(lang for p in group for lang in (p[0]['languages'] or ['unknown']))
        for lang in ['all'] + [lang for lang, n in languages.most_common() if n >= 3]:
            target = group if lang == 'all' else [p for p in group if lang in (p[0]['languages'] or ['unknown'])]
            for scope, index in (('page', 1), ('headings', 2)):
                docs = defaultdict(set)
                for p in target:
                    for w in p[index]:
                        docs[w].add(p[0]['top'])
                top_docs = Counter({w: len(d) for w, d in docs.items()})
                out.append(f'- **{lang}, {scope}** ({len(target)}): '
                           + table([p[index] for p in target], [p[index] for p in others], top_docs))
            if lang != 'all' and lang != 'ces':
                for p in target[:args.headings]:
                    out.append(f'  - `{p[3]}`')
        out.append('')
    path = args.work_dir / 'dev' / 'keyword_candidates.md'
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
