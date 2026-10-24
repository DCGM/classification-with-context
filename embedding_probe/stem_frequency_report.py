"""Development tool: how often each keyword stem of the page statistics fires, on PUBLIC text only, to
decide which stems to keep.

  random   --random pages drawn uniformly (by a hash of the key) from the public text LMDB: the share
           of pages with a hit anywhere and in the header (first 3 lines)
  labelled the DB-labelled public pages of the dev samples (dev_pages.parquet, keyword_sample.jsonl):
           the page type on whose pages the stem fires most often (rate) and how much more often
           than on all labelled pages (lift); and the hit rate on the page types of its group

The private text LMDB is never opened. Writes a markdown table per group (and the stems that never fire)
to --out.

Example:
    python -m embedding_probe.stem_frequency_report --out embeddings/text/page_stats/stem_frequencies.md
"""
import argparse
import hashlib
import json
from pathlib import Path

import lmdb
import numpy as np
import polars as pl

from embedding_probe.common import PUBLIC_TEXT_LMDB, WORK_DIR
from embeddings.text.page_stats.page_stats import FEATURES, KEYWORDS, STEMS, PageStats

# the DB page types a group is meant to spot (for the 'on own types' column)
GROUP_TYPES = {
    'toc': ('TableOfContents',), 'index': ('Index',), 'errata': ('Errata',), 'preface': ('Preface', 'introduction'),
    'dedication': ('Dedication',), 'imprimatur': ('Imprimatur',), 'imprint': ('Impressum', 'colophon'),
    'bibliography': ('Bibliography',), 'list_illustrations': ('ListOfIllustrations',), 'list_maps': ('ListOfMaps',),
    'list_tables': ('ListOfTables',), 'appendix': ('Appendix',), 'abstract': ('Abstract',),
    'advertisement': ('Advertisement',), 'title': ('TitlePage',), 'masthead': ('TitlePage',),
    'calibration': ('CalibrationTable',), 'music': ('SheetMusic',), 'map': ('Map',), 'stamp': (),
    'table': ('Table',), 'cover': ('FrontCover', 'BackCover', 'Cover', 'Jacket', 'FrontJacket'),
    'end': ('Impressum', 'colophon'),
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--random', type=int, default=30000)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args(argv)


def random_keys(env, n: int, total: int):
    """About n keys, chosen by a hash of the key (uniform over the LMDB)."""
    threshold = int(2 ** 64 * min(1.0, n / total))
    keys = []
    with env.begin() as txn, txn.cursor() as cursor:
        for key in cursor.iternext(values=False):
            if int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little') < threshold:
                keys.append(key)
    return keys


def main(argv=None):
    args = parse_arguments(argv)
    stats = PageStats()
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    total = env.stat()['entries']
    keys = random_keys(env, args.random, total)
    labelled = pl.read_parquet(args.work_dir / 'dev' / 'dev_pages.parquet').select('key', 'db_type')
    sample = [json.loads(line) for line in open(args.work_dir / 'dev' / 'keyword_sample.jsonl', encoding='utf-8')]
    labelled = pl.concat([labelled, pl.DataFrame({'key': [r['key'] for r in sample],
                                                  'db_type': [r['db_type'] for r in sample]})]).unique('key')
    with env.begin() as txn:
        random_texts = [txn.get(k).decode('utf-8') for k in keys]
        labelled_texts = [txn.get(k.encode()).decode('utf-8') for k in labelled['key']]
    col = {n: i for i, n in enumerate(FEATURES)}
    page_cols = [col[f'kw:{s}'] for s in STEMS]
    head_cols = [col[f'head_kw:{s}'] for s in STEMS]
    r = stats.embed(random_texts, kind='text', workers=args.workers)
    r_page, r_head = (r[:, page_cols] > 0).mean(0), (r[:, head_cols] > 0).mean(0)
    del r
    lab = stats.embed(labelled_texts, kind='text', workers=args.workers)[:, page_cols] > 0
    types = labelled['db_type'].to_numpy()
    type_names = sorted(set(types))
    type_rate = np.stack([lab[types == t].mean(0) for t in type_names], 1)  # stems x types
    overall = lab.mean(0)
    counts = {t: int((types == t).sum()) for t in type_names}

    def best_type(i):
        if overall[i] == 0:
            return '-', '-'
        valid = [j for j, t in enumerate(type_names) if counts[t] >= 30]
        j = max(valid, key=lambda j: type_rate[i, j])
        return f'{type_names[j]} {type_rate[i, j]:.0%}', f'{type_rate[i, j] / overall[i]:.0f}x'

    def own_rate(i, group):
        mask = np.isin(types, GROUP_TYPES[group])
        return f'{lab[mask, i].mean():.0%}' if mask.any() else '-'

    index = {s: i for i, s in enumerate(STEMS)}
    out = ['# Keyword stem frequencies (public text only)', '',
           f'Random: {len(keys)} pages drawn uniformly from the {total} public pages; share of pages with a hit '
           f'anywhere / in the first 3 lines. Labelled: {len(labelled)} DB-labelled public pages of the dev samples '
           '(skewed to the rare page types); the page type with the highest hit rate (types with 30+ pages), '
           'its lift over all labelled pages, and the hit rate on the group\'s own page types. '
           'Public text is mostly Czech and German: a stem of another language that never fires here is untested, '
           'not useless.', '']
    never = []
    for group, by_language in KEYWORDS.items():
        out += [f'## {group} (own types: {", ".join(GROUP_TYPES[group]) or "none"})', '',
                '| stem | languages | random page | random header | labelled: top type | lift | own types |',
                '|---|---|---:|---:|---|---:|---:|']
        seen = {}
        for lang, stems in by_language.items():
            for s in stems:
                seen.setdefault(s, []).append(lang)
        for s, langs in seen.items():
            i = index[s]
            top, lift = best_type(i)
            out.append(f'| `{s}` | {", ".join(langs)} | {r_page[i]:.2%} | {r_head[i]:.2%} | {top} | {lift} | '
                       f'{own_rate(i, group)} |')
            if r_page[i] == 0 and overall[i] == 0:
                never.append((group, s, langs))
        out.append('')
    out += ['## Stems that never fired on the public samples', '']
    by_lang = {}
    for group, s, langs in never:
        for lang in langs:
            by_lang.setdefault(lang, []).append(f'`{s}` ({group})')
    for lang, items in sorted(by_lang.items(), key=lambda x: -len(x[1])):
        out.append(f'- **{lang}** ({len(items)}): ' + ', '.join(items))
    args.out.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'{len(keys)} random, {len(labelled)} labelled pages; {len(never)} stems never fired; wrote {args.out}')


if __name__ == '__main__':
    main()
