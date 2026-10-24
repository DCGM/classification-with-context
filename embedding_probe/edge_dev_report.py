"""Development report: how to find a page's header and footer lines for the text cues, on PUBLIC pages
with their original ALTO (the DB-labelled sample of build_dev_sample.py; tst untouched).

For each method of ocr_layout.header_footer_counts (first/last 3 lines, vertical gaps, bands of the
text box, gaps + tall lines):
  1. how many lines the header and footer have (median),
  2. spotting: per keyword group, the share of its target page type's pages on which the group fires
     in the header (footer for imprints) vs the share of all other pages (false alarms),
  3. a classifier on the text cues (with that header/footer) + the ALTO line features, trained on 75%
     of the sample documents and scored on the rest (weighted F1, DB labels).

Example:
    python -m embedding_probe.edge_dev_report
"""
import argparse
import hashlib
import zlib
from pathlib import Path

import lmdb
import numpy as np
import polars as pl

from embedding_probe.common import PAGE_TYPES, PUBLIC_TEXT_LMDB, WORK_DIR, evaluate, label_ids
from embedding_probe.ocr_layout import (EDGE_METHODS, LAYOUT_XML_FEATURES, LINE_FEATURES, header_footer_counts,
                                        layout_vector, ocr_text_lines)
from embedding_probe.text_cues import CUE_FEATURES, cue_matrix

TARGETS = {'toc': 'TableOfContents', 'index': 'Index', 'errata': 'Errata', 'preface': 'Preface',
           'dedication': 'Dedication', 'imprimatur': 'Imprimatur', 'bibliography': 'Bibliography',
           'list_illustrations': 'ListOfIllustrations', 'list_maps': 'ListOfMaps', 'list_tables': 'ListOfTables',
           'appendix': 'Appendix'}
FOOT_TARGETS = {'imprint': 'Impressum', 'end': 'Impressum'}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier

    pages = pl.read_parquet(args.work_dir / 'dev' / 'dev_pages.parquet')
    env = lmdb.open(str(PUBLIC_TEXT_LMDB.parent / 'alto'), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        xmls = [zlib.decompress(txn.get(k.encode('utf-8'))) for k in pages['key']]
    env.close()
    texts = ['\n'.join(ocr_text_lines(x)) for x in xmls]
    layout_col = {n: i for i, n in enumerate(LAYOUT_XML_FEATURES)}
    layout = np.array([layout_vector(x) for x in xmls], dtype=np.float32)[:, [layout_col[n] for n in LINE_FEATURES]]
    labels = pages['db_type'].to_numpy()
    y = label_ids(labels.tolist())
    col = {n: i for i, n in enumerate(CUE_FEATURES)}
    doc_hash = np.array([int.from_bytes(hashlib.blake2b(f'{lib}/{doc}'.encode(), digest_size=8).digest(), 'little')
                         / 2 ** 64 for lib, doc in pages.select('library', 'doc').iter_rows()])
    test = doc_hash < 0.25
    print(f'{len(pages)} public pages with ALTO, {test.sum()} held out\n')

    spotting, summary = {}, {}
    for method in EDGE_METHODS:
        counts = [header_footer_counts(x, method) for x in xmls]
        cues = cue_matrix([(t, h, f) for t, (h, f) in zip(texts, counts)])
        heads, foots = np.array([c[0] for c in counts]), np.array([c[1] for c in counts])
        spotting[method] = {}
        for prefix, targets in (('head', TARGETS), ('foot', FOOT_TARGETS)):
            for group, target in targets.items():
                fired = cues[:, col[f'{prefix}_{group}']] > 0
                own = labels == target
                spotting[method][f'{prefix} {group} ({target})'] = (fired[own].mean(), fired[~own].mean())
        x = np.hstack([cues, layout])
        model = HistGradientBoostingClassifier(max_iter=300, early_stopping=False, class_weight='balanced',
                                               random_state=0).fit(x[~test], y[~test])
        score = evaluate(labels[test].tolist(), [PAGE_TYPES[i] for i in model.predict(x[test])])
        summary[method] = (np.median(heads), np.median(foots), score['weighted_fscore'], score['accuracy'])

    print('## Header and footer size, classifier (text cues + ALTO line features)\n')
    print('| Method | median header lines | median footer lines | weighted F1 | accuracy |\n|---|---:|---:|---:|---:|')
    for method, (h, f, wf, acc) in summary.items():
        print(f'| {method} | {h:.0f} | {f:.0f} | {wf:.3f} | {acc:.3f} |')
    print('\n## Spotting: recall on the target type / false alarms on all other pages\n')
    print('| Group in header/footer | ' + ' | '.join(EDGE_METHODS) + ' |\n|---|' + '---|' * len(EDGE_METHODS))
    for name in spotting[EDGE_METHODS[0]]:
        print(f'| {name} | ' + ' | '.join(f'{spotting[m][name][0]:.0%} / {spotting[m][name][1]:.1%}'
                                          for m in EDGE_METHODS) + ' |')


if __name__ == '__main__':
    main()
