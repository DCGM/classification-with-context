"""Scores the predictions with the README metrics and writes the reports.

For every context window K with predictions (<work-dir>/k<K>/predictions/*.parquet) it writes
<work-dir>/k<K>/report.md and metrics.json, and over all K <work-dir>/report.md, the weighted F1 of
every experiment per K.

Metrics (see "Page type classification evaluation" in the README): accuracy, weighted precision,
recall and F1 over all class ids, and per-class F1. For K > 0 the tst pages are also split by
whether their window is complete (full) or runs past the document's start (left cut), end (right
cut) or both. A prediction file that covers only part of tst (e.g. text_baselines.public_only) is
scored on its pages, and the complete experiments are scored on the same pages next to it.

Example:
    python -m embedding_probe.report
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

from embedding_probe.common import PAGE_TYPES, WORK_DIR, evaluate, setup_logging, window_completeness

logger = logging.getLogger('REPORT')
PER_CLASS_COLUMNS = 6
COMPLETENESS = ('full', 'left cut', 'right cut', 'both cut')
DB_SUFFIX = ' (DB labels)'
CUE_PAIRS = (('mmBERT+DINOv2', 'mmBERT+DINOv2 +cues'), ('layout stats', 'text cues'))


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    return parser.parse_args(argv)


def load_predictions(pred_dir: Path, tst_keys: List[str]) -> Dict[str, pl.DataFrame]:
    """File stem -> predictions (key, label, experiment columns), each checked against the tst pages."""
    files = {}
    tst = set(tst_keys)
    for path in sorted(pred_dir.glob('*.parquet')):
        frame = pl.read_parquet(path)
        assert set(frame['key']) <= tst and frame['key'].n_unique() == len(frame), f'{path.name}: bad keys'
        files[path.stem] = frame
    return files


def score_all(frame: pl.DataFrame) -> Dict[str, Dict]:
    return {e: evaluate(frame['label'].to_list(), frame[e].fill_null('').to_list())
            for e in frame.columns if e not in ('key', 'label', 'completeness')}


def summary_table(scores: Dict[str, Dict]) -> List[str]:
    rows = ['| Experiment | Accuracy | Weighted P | Weighted R | Weighted F1 |', '|---|---:|---:|---:|---:|']
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]['weighted_fscore']):
        rows.append(f'| {name} | {s["accuracy"]:.3f} | {s["weighted_precision"]:.3f} | '
                    f'{s["weighted_recall"]:.3f} | {s["weighted_fscore"]:.3f} |')
    return rows


def per_class_table(scores: Dict[str, Dict], columns: List[str]) -> List[str]:
    rows = ['| Page type | tst pages | ' + ' | '.join(columns) + ' |', '|---|---:|' + '---:|' * len(columns)]
    support = scores[columns[0]]['per_class']
    for c in sorted(support, key=lambda c: (-support[c]['support'], c)):
        values = [scores[e]['per_class'].get(c, {}).get('fscore', 0.0) for e in columns]
        best = max(values)
        cells = [f'**{v:.2f}**' if v == best and v > 0 else f'{v:.2f}' for v in values]
        rows.append(f'| {c} | {support[c]["support"]} | ' + ' | '.join(cells) + ' |')
    return rows


def completeness_table(full: pl.DataFrame, experiments: List[str]) -> List[str]:
    groups = [g for g in COMPLETENESS if (full['completeness'] == g).any()]
    counts = {g: int((full['completeness'] == g).sum()) for g in groups}
    rows = ['| Experiment | ' + ' | '.join(f'{g} ({counts[g]})' for g in groups) + ' |',
            '|---|' + '---:|' * len(groups)]
    for e in experiments:
        cells = []
        for g in groups:
            part = full.filter(pl.col('completeness') == g)
            cells.append(f'{evaluate(part["label"].to_list(), part[e].fill_null("").to_list())["accuracy"]:.3f}')
        rows.append(f'| {e} | ' + ' | '.join(cells) + ' |')
    return rows


def annotations_vs_db(scores: Dict[str, Dict]) -> List[str]:
    """Each experiment trained on the annotations next to the same experiment trained on the DB page types."""
    pairs = [e for e in scores if e + DB_SUFFIX in scores]
    if not pairs:
        return []
    pairs.sort(key=lambda e: -scores[e]['weighted_fscore'])
    lines = ['', '### Trained on the annotations vs on the DB page types', '',
             'The same experiment trained on the annotated page types of the trn pages, or on their DB page '
             'types (the pages without one left out); both scored on the annotated tst pages. The gain is '
             'what the annotators\' corrections bring.', '',
             '| Experiment | Weighted F1, annotations | Weighted F1, DB types | Gain |', '|---|---:|---:|---:|']
    for e in pairs:
        a, b = scores[e]['weighted_fscore'], scores[e + DB_SUFFIX]['weighted_fscore']
        lines.append(f'| {e} | {a:.3f} | {b:.3f} | {a - b:+.3f} |')
    best = pairs[0]
    a, b = scores[best]['per_class'], scores[best + DB_SUFFIX]['per_class']
    gain = {c: a[c]['fscore'] - b.get(c, {}).get('fscore', 0.0) for c in a}
    lines += ['', f'Per-class F1 of {best}, ordered by the gain:', '',
              '| Page type | tst pages | Annotations | DB types | Gain |', '|---|---:|---:|---:|---:|']
    for c in sorted(gain, key=lambda c: (-gain[c], c)):
        lines.append(f'| {c} | {a[c]["support"]} | {a[c]["fscore"]:.2f} | '
                     f'{b.get(c, {}).get("fscore", 0.0):.2f} | {gain[c]:+.2f} |')
    return lines


def pair_section(scores: Dict[str, Dict], base: str, variant: str) -> List[str]:
    """Per-class F1 of an experiment and of its variant (e.g. with the text cues), ordered by the gain."""
    if base not in scores or variant not in scores:
        return []
    a, b = scores[base], scores[variant]
    lines = ['', f'### {variant} vs {base}', '',
             f'Weighted F1 {b["weighted_fscore"]:.3f} vs {a["weighted_fscore"]:.3f} '
             f'({b["weighted_fscore"] - a["weighted_fscore"]:+.3f}); per class, ordered by the gain:', '',
             f'| Page type | tst pages | {base} | {variant} | Gain |', '|---|---:|---:|---:|---:|']
    gain = {c: b['per_class'].get(c, {}).get('fscore', 0.0) - v['fscore'] for c, v in a['per_class'].items()}
    for c in sorted(gain, key=lambda c: (-gain[c], c)):
        lines.append(f'| {c} | {a["per_class"][c]["support"]} | {a["per_class"][c]["fscore"]:.2f} | '
                     f'{b["per_class"].get(c, {}).get("fscore", 0.0):.2f} | {gain[c]:+.2f} |')
    return lines


def report_context(work_dir: Path, pages: pl.DataFrame, context: int) -> Dict[str, Dict]:
    tst = pages.filter(pl.col('split') == 'tst')
    files = load_predictions(work_dir / f'k{context}' / 'predictions', tst['key'].to_list())
    complete = {stem: f for stem, f in files.items() if len(f) == len(tst)}
    partial = {stem: f for stem, f in files.items() if len(f) < len(tst)}

    full = tst.select('key', 'label').with_columns(
        pl.Series('completeness', window_completeness(tst, context).tolist()))
    for frame in complete.values():
        full = full.join(frame.drop('label'), on='key', how='left', maintain_order='left')
    scores = score_all(full)
    ranked = sorted(scores, key=lambda e: -scores[e]['weighted_fscore'])

    lines = [f'# Page type probes, context window K = {context}', '',
             f'Trained on the annotated trn pages, scored on the {len(tst)} annotated tst pages '
             f'(metrics as in the README: weighted over all {len(PAGE_TYPES)} class ids). K = {context}: '
             f'{"the page alone" if context == 0 else f"the page and {context} page(s) on each side"}. '
             f'Prediction files: {", ".join(sorted(files))}.', '', '## All tst pages', '']
    lines += summary_table(scores)
    columns = [e for e in ranked if not e.endswith(DB_SUFFIX)][:PER_CLASS_COLUMNS]
    if 'DB page type' in scores and 'DB page type' not in columns:
        columns.append('DB page type')
    lines += ['', '### Per-class F1 of the best experiments', ''] + per_class_table(scores, columns)
    lines += annotations_vs_db(scores)
    for base, variant in CUE_PAIRS:
        lines += pair_section(scores, base, variant)
    if context > 0:
        lines += ['', '### Accuracy by window completeness', '',
                  'Whether the window of a tst page reaches past its document\'s start (left cut), end '
                  '(right cut) or both; the page count in brackets.', '']
        lines += completeness_table(full, [e for e in ranked if not e.endswith(DB_SUFFIX)])
    metrics = {'all tst pages': scores}

    for stem, frame in partial.items():
        subset = frame.select('key', 'label')
        for other in complete.values():
            subset = subset.join(other.drop('label'), on='key', how='left', maintain_order='left')
        subset = subset.join(frame.drop('label'), on='key', how='left', maintain_order='left')
        sub_scores = score_all(subset)
        lines += ['', f'## The {len(frame)} tst pages of {stem}', '',
                  f'{stem} covers only these tst pages; all experiments are scored on them here.', '']
        lines += summary_table(sub_scores)
        metrics[f'tst pages of {stem}'] = sub_scores

    (work_dir / f'k{context}' / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (work_dir / f'k{context}' / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    logger.info(f'wrote {work_dir / f"k{context}" / "report.md"}')
    return scores


def main(argv=None):
    args = parse_arguments(argv)
    setup_logging('REPORT')
    pages = pl.read_parquet(args.work_dir / 'pages.parquet')
    contexts = sorted(int(m.group(1)) for d in args.work_dir.glob('k*/predictions')
                      if (m := re.fullmatch(r'k(\d+)', d.parent.name)) and any(d.glob('*.parquet')))
    by_context = {k: report_context(args.work_dir, pages, k) for k in contexts}

    experiments = list(dict.fromkeys(e for scores in by_context.values() for e in scores))
    last = by_context[contexts[-1]]
    experiments.sort(key=lambda e: -last.get(e, {}).get('weighted_fscore', -1))
    lines = ['# Page type probes: weighted F1 by context window K', '',
             'K pages on each side of the page (K = 0: the page alone); all annotated tst pages. Details '
             'per K in k<K>/report.md.', '',
             '| Experiment | ' + ' | '.join(f'K = {k}' for k in contexts) + ' |', '|---|' + '---:|' * len(contexts)]
    for e in experiments:
        values = [by_context[k][e]['weighted_fscore'] if e in by_context[k] else np.nan for k in contexts]
        lines.append(f'| {e} | ' + ' | '.join('' if np.isnan(v) else f'{v:.3f}' for v in values) + ' |')
    (args.work_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logger.info(f'wrote {args.work_dir / "report.md"}')


if __name__ == '__main__':
    main()
