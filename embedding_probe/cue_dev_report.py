"""Development report of the text cues (text_cues.py) on PUBLIC text only, without touching tst.

Data: the DB-labelled public sample of build_dev_sample.py (trn context documents; label = DB page
type). Reads only the public text LMDB.

  1. Spotting: per keyword group, the share of the pages of each page type on which the group fires
     in the headings (first 3 lines) and anywhere; the types with the highest rates are listed.
  2. Classifier: gradient boosting on the 17 layout statistics, the text cues (on the text LMDB's
     line order), the OCR XML layout features (ocr_layout.py, from PAGE XML and, where fetched, ALTO)
     and the cues on the XML lines sorted top to bottom + the layout features together, trained
     on 75% of the sample documents and scored on the other 25% (per-class F1). Labels are DB types,
     so this is for comparing the two feature sets, not an absolute score.

Example:
    python -m embedding_probe.cue_dev_report
"""
import argparse
import hashlib
from pathlib import Path

import lmdb
import numpy as np
import polars as pl

from embedding_probe.common import PUBLIC_TEXT_LMDB, WORK_DIR, evaluate, label_ids
from embedding_probe.text_baselines import LAYOUT_FEATURES
from embedding_probe.ocr_layout import layout_vector, ocr_text_lines
from embedding_probe.text_cues import CUE_FEATURES, KEYWORDS, cue_matrix

PUBLIC_OCR_LMDBS = {'PAGE XML': PUBLIC_TEXT_LMDB.parent / 'page_xml', 'ALTO': PUBLIC_TEXT_LMDB.parent / 'alto'}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--top', type=int, default=4, help='Page types listed per keyword group.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier

    pages = pl.read_parquet(args.work_dir / 'dev' / 'dev_pages.parquet')
    env = lmdb.open(str(PUBLIC_TEXT_LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        texts = [txn.get(k.encode('utf-8')).decode('utf-8') for k in pages['key']]
    env.close()
    labels = pages['db_type'].to_numpy()
    x = cue_matrix(texts)
    import zlib
    layouts, xml_cues = {}, {}
    for kind, path in PUBLIC_OCR_LMDBS.items():
        if not path.exists():
            continue
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            xmls = [txn.get(k.encode('utf-8')) for k in pages['key']]
        env.close()
        xmls = [zlib.decompress(a) if a is not None else None for a in xmls]
        layouts[kind] = np.array([layout_vector(a) for a in xmls], dtype=np.float32)
        xml_cues[kind] = cue_matrix(['\n'.join(ocr_text_lines(a)) if a is not None else '' for a in xmls])
        print(f'{sum(a is not None for a in xmls)} pages with {kind}')
    col = {name: i for i, name in enumerate(CUE_FEATURES)}

    print(f'{len(pages)} public DB-labelled pages\n\n## Spotting: share of the pages of a type on which a group fires\n')
    print('| Group | Headings: top types | Anywhere: top types |\n|---|---|---|')
    classes = sorted(set(labels))
    for group in KEYWORDS:
        cells = []
        for prefix in ('head', 'kw'):
            fired = x[:, col[f'{prefix}_{group}']] > 0
            rates = sorted(((fired[labels == c].mean(), c) for c in classes), reverse=True)[:args.top]
            cells.append(', '.join(f'{c} {r:.0%}' for r, c in rates if r > 0))
        print(f'| {group} | {cells[0]} | {cells[1]} |')

    doc_hash = np.array([int.from_bytes(hashlib.blake2b(f'{lib}/{doc}'.encode(), digest_size=8).digest(), 'little')
                         / 2 ** 64 for lib, doc in pages.select('library', 'doc').iter_rows()])
    test = doc_hash < 0.25
    y = label_ids(labels.tolist())
    layout_cols = [col[f] for f in LAYOUT_FEATURES]
    scores = {}
    feature_sets = [('layout stats', x[:, layout_cols]), ('text cues', x)]
    for kind in layouts:
        feature_sets += [(f'{kind} layout', layouts[kind]),
                         (f'{kind}-order cues + {kind} layout', np.hstack([xml_cues[kind], layouts[kind]]))]
    for name, features in feature_sets:
        model = HistGradientBoostingClassifier(max_iter=300, early_stopping=False, class_weight='balanced',
                                               random_state=0).fit(features[~test], y[~test])
        scores[name] = evaluate(labels[test].tolist(), [classes_name(i) for i in model.predict(features[test])])
    print(f'\n## Classifier on {test.sum()} held-out pages (DB labels)\n')
    print('| | ' + ' | '.join(scores) + ' |\n|---|' + '---:|' * len(scores))
    print('| weighted F1 | ' + ' | '.join(f'{s["weighted_fscore"]:.3f}' for s in scores.values()) + ' |')
    print('| accuracy | ' + ' | '.join(f'{s["accuracy"]:.3f}' for s in scores.values()) + ' |')
    per = scores[feature_sets[-1][0]]['per_class']
    for c in sorted(per, key=lambda c: -per[c]['fscore']):
        print(f'| {c} ({per[c]["support"]}) | ' + ' | '.join(
            f'{s["per_class"].get(c, {}).get("fscore", 0):.2f}' for s in scores.values()) + ' |')


def classes_name(i: int) -> str:
    from embedding_probe.common import PAGE_TYPES
    return PAGE_TYPES[i]


if __name__ == '__main__':
    main()
