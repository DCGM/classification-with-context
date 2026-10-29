"""Compare two MetaKat engines on the prepared batches against the reference labels of the manifests.

Page type: the curated (annotated) label where a page has one, else the DB page type. Side: the DB placement,
where the page has one. Document starts: the true document starts of the batch (boundaries are known); the first
page of a batch is never predicted. Pages whose reference page type is outside the label set of both engines are
left out. Scores per set and group, and for the curated pages alone, with a paired McNemar test of the
page-type correctness of the second engine against the first.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support

SIDES = {'left': 'left', 'right': 'right', 'single': 'singlePage'}


def type_name(label):
    return 'flyleaf' if label == 'FlyLeaf' else label[:1].lower() + label[1:]


def engine_types(config_path):
    labels = json.loads(config_path.read_text())['definition']['page_type']['core'].get('labels')
    return None if labels is None else set(labels)


def load_pages(root, engines):
    """One row per page: reference labels and, per engine, its page type, side and start-of-document."""
    rows = []
    for batch in sorted(p for p in root.iterdir() if p.is_dir() and (p / 'READY.json').is_file()):
        manifest = json.loads((batch / 'manifest.json').read_text())
        predicted = {}
        for engine in engines:
            result = json.loads((batch.with_name(f'{batch.name}.{engine}.results') / 'metakat.json').read_text())
            predicted[engine] = {p['id']: p for p in result['elements'] if p['type'] == 'page'}
        for position, page in enumerate(manifest['pages']):
            curated = page.get('annotated_type') or ''
            row = {'set': manifest['set'], 'group': manifest['group'], 'batch': batch.name, 'page': page['page'],
                   'position': position, 'curated': bool(curated),
                   'type': type_name(curated or page.get('db_type') or '') if (curated or page.get('db_type')) else None,
                   'side': SIDES.get(page.get('placement') or ''), 'start': bool(page['document_start'])}
            for engine in engines:
                p = predicted[engine][page['page']]
                starts = p.get('startsDocument')
                row[engine] = {'type': p['pageType']['text'], 'side': (p.get('side') or {}).get('text'),
                               'start': None if starts is None else bool(starts['val'])}
            rows.append(row)
    return rows


def classification(truth, prediction):
    labels = sorted(set(truth))
    accuracy = float(np.mean([t == p for t, p in zip(truth, prediction)]))
    weighted = precision_recall_fscore_support(truth, prediction, labels=labels, average='weighted',
                                               zero_division=0)[2]
    macro = precision_recall_fscore_support(truth, prediction, labels=labels, average='macro', zero_division=0)[2]
    return {'n': len(truth), 'accuracy': accuracy, 'weighted_f1': float(weighted), 'macro_f1': float(macro)}


def per_class(truth, predictions, labels):
    """Support, precision, recall and F1 of each reference label, for each engine's predictions."""
    table = {}
    for engine, prediction in predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(truth, prediction, labels=labels,
                                                                        zero_division=0)
        table[engine] = {label: {'support': int(support[i]), 'precision': float(precision[i]),
                                 'recall': float(recall[i]), 'f1': float(f1[i])} for i, label in enumerate(labels)}
    return table


def boundary_group(row):
    """The batch group, the long boundaries batches split by what they hold (issues, books or a mixture)."""
    return f'boundaries/{row["batch"].split("_")[2]}' if row['group'] == 'boundaries' else row['group']


def boundary_scores(truth, prediction):
    truth, prediction = np.array(truth), np.array(prediction)
    tp = int((truth & prediction).sum())
    precision = tp / max(int(prediction.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    return {'pages': len(truth), 'true_starts': int(truth.sum()), 'predicted_starts': int(prediction.sum()),
            'hits': tp, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-12)}


def percent(value):
    return '-' if value is None else f'{100 * value:.1f}'


def render_markdown(report, engines):
    first, second = engines
    out = [f'# MetaKat engine comparison: {first} vs {second}', '',
           f'**{first}** predicts the page type of each page alone and has no side or document-start model of its '
           f'own. **{second}** reads the whole batch as one scan (512-page windows) and predicts page type, side '
           'and document starts together. Both ran over the same prepared batches, with the same images and ALTO.',
           '', '## Reference labels', '',
           '- **Page type:** the curated (annotated) label where the page has one, otherwise the DB page type. '
           'DB types are partly weak labels, so absolute scores understate both engines; the curated rows are the '
           'cleanest.',
           '- **Side:** the DB placement (`left`, `right`, `single`), which exists for few pages only.',
           '- **Document starts:** the true document boundaries of each batch (known from the DB hierarchy). The '
           'first page of a batch is not predicted and not scored.',
           '- **Sets:** `evaluation` documents come from the dev split, so its results are held-out. `consistency` '
           'documents come from the fit split, which the context model may have been trained on: its '
           f'{second} numbers are inflated and shown only as a sanity check.',
           '- Percentages everywhere; *fixed* = pages the second engine gets right and the first gets wrong, '
           '*broken* = the reverse (for the page type or side of the reference class).', '']
    evaluation = report['sets']['evaluation']
    boundary = evaluation['boundary']['all']
    out += ['## Summary (evaluation set, held-out)', '',
            f'- Page type accuracy over {evaluation["type"]["all"][first]["n"]} pages: '
            f'{percent(evaluation["type"]["all"][first]["accuracy"])} for {first}, '
            f'{percent(evaluation["type"]["all"][second]["accuracy"])} for {second}; on the curated pages '
            f'{percent(evaluation["type"]["curated"][first]["accuracy"])} and '
            f'{percent(evaluation["type"]["curated"][second]["accuracy"])}.',
            f'- Document starts: F1 {percent(boundary[first]["f1"])} for {first}, {percent(boundary[second]["f1"])} '
            f'for {second} ({boundary[second]["hits"]} of {boundary[second]["true_starts"]} starts found).',
            f'- Side ({evaluation["side"]["n"]} pages with a DB placement): '
            f'{percent(evaluation["side"][second])} for {second}; {first} has none.']
    worse = [(label, evaluation['per_type']['table'][first][label], evaluation['per_type']['table'][second][label])
             for label in evaluation['per_type']['labels']
             if evaluation['per_type']['table'][first][label]['support'] >= 20 and
             evaluation['per_type']['table'][second][label]['f1'] < evaluation['per_type']['table'][first][label]['f1']
             - 0.05]
    if worse:
        out.append(f'- Page types with at least 20 reference pages where {second} has a lower F1 than {first}: ' +
                   ', '.join(f'{label} ({percent(a["f1"])} → {percent(b["f1"])})' for label, a, b in worse) + '.')
    out.append('')
    for set_name, section in report['sets'].items():
        held = ' (held-out, the clean comparison)' if set_name == 'evaluation' else ' (fit documents: not held-out)'
        out += [f'## {set_name.capitalize()} set{held}', '',
                f'{section["pages"]} pages in {sum(section["batches_by_group"].values())} batches; '
                f'{section["without_type"]} pages have no reference page type and are not scored for it.', '']
        allscores = section['type']['all']
        out += ['### Page type, all pages together', '',
                '| | Pages | Accuracy | Weighted F1 | Macro F1 |', '|---|---:|---:|---:|---:|']
        for engine in engines:
            m = allscores[engine]
            out.append(f'| {engine} | {m["n"]} | {percent(m["accuracy"])} | {percent(m["weighted_f1"])} | '
                       f'{percent(m["macro_f1"])} |')
        out += ['', f'{second} fixes {allscores["second_right_first_wrong"]} pages and breaks '
                    f'{allscores["first_right_second_wrong"]} (paired McNemar p = {allscores["mcnemar_p"]:.2g}).']
        curated = section['type'].get('curated')
        if curated:
            out += [f'On the {curated[first]["n"]} curated pages alone: {first} accuracy '
                    f'{percent(curated[first]["accuracy"])}, {second} {percent(curated[second]["accuracy"])} '
                    f'(fixed {curated["second_right_first_wrong"]}, broken {curated["first_right_second_wrong"]}).']
        table = section['per_type']['table']
        out += ['', '#### Per page type', '',
                f'Precision (P), recall (R) and F1 per engine; *fixed*/*broken* count the pages of that reference '
                f'class that {second} gets right or wrong where {first} did not or did.', '',
                f'| Page type | Support | {first} P | R | F1 | {second} P | R | F1 | Fixed | Broken |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label in section['per_type']['labels']:
            a, b, c = table[first][label], table[second][label], table[label]
            out.append(f'| {label} | {a["support"]} | {percent(a["precision"])} | {percent(a["recall"])} | '
                       f'{percent(a["f1"])} | {percent(b["precision"])} | {percent(b["recall"])} | '
                       f'{percent(b["f1"])} | {c["fixed"]} | {c["broken"]} |')
        out += ['', '#### Most frequent page-type errors (reference → predicted)', '']
        for engine in engines:
            out.append(f'- **{engine}:** ' + ', '.join(f'{t} → {p} ({n})'
                                                       for (t, p), n in section['per_type']['confusions'][engine]))
        side = section['per_side']
        n_side = section['side']['n']
        out += ['', f'### Side, all pages together ({n_side} pages with a DB placement)', '',
                f'{first} has no side prediction. Precision is computed over the pages that have a reference '
                'placement only.', '',
                f'| Side | Support | {second} P | R | F1 |', '|---|---:|---:|---:|---:|']
        for label in side['labels']:
            m = side['table'][second][label]
            out.append(f'| {label} | {m["support"]} | {percent(m["precision"])} | {percent(m["recall"])} | '
                       f'{percent(m["f1"])} |')
        out += ['', f'Overall side accuracy of {second}: {percent(section["side"][second])}. '
                    'Confusion (reference → predicted): ' +
                ', '.join(f'{pair}: {n}' for pair, n in sorted(side['confusion'][second].items())), '']
        out += ['### Document starts, by kind of batch', '',
                'The long `boundaries/*` batches hold many short documents next to each other: `issues` (periodical '
                'issues of one volume), `books` (unrelated short books) and `mixed`. The others are the regular '
                'groups: `issues` and `multipart` batches of two or three adjacent documents, `monographs` of '
                'unrelated standalone books, `single` of one document (no true starts to find, so every predicted '
                'start there is a false alarm).', '',
                '| Batch kind | Batches | Pages scored | True starts | ' +
                ' | '.join(f'{e}: found | predicted | P | R | F1' for e in engines) + ' |',
                '|---|---:|---:|---:|' + '---:|---:|---:|---:|---:|' * len(engines)]
        for name, scores in section['boundary'].items():
            m1 = scores[first]
            batches = section['batches_by_group'].get(name, sum(section['batches_by_group'].values())
                                                      if name == 'all' else 0)
            cells = []
            for engine in engines:
                m = scores[engine]
                cells.append('- | - | - | - | -' if m is None else
                             f'{m["hits"]} | {m["predicted_starts"]} | {percent(m["precision"])} | '
                             f'{percent(m["recall"])} | {percent(m["f1"])}')
            reference = m1 or scores[second]
            out.append(f'| {name} | {batches} | {reference["pages"]} | {reference["true_starts"]} | ' +
                       ' | '.join(cells) + ' |')
        out.append('')
    return '\n'.join(out) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types'))
    parser.add_argument('--engines', nargs=2, default=['chapter-v1.6.0', 'chapter-v1.7.0'])
    parser.add_argument('--engine-configs', nargs=2, type=Path, default=[
        Path('/mnt/kolosus/data/metakat_worker/src_engines/chapter.v1.6.0.engine.json'),
        Path('/mnt/kolosus/data/metakat_worker/src_engines/chapter.v1.7.0.engine.json')])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, help='Output directory of page_sequence.evaluate_conditions; its '
                                                         'evaluation is appended to the Markdown report.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    first, second = args.engines
    vocabulary = engine_types(args.engine_configs[0])
    vocabulary = None if vocabulary is None else {type_name(v) for v in vocabulary}
    report = {'engines': args.engines, 'sets': {}}
    csv_rows = []
    for set_name in ('evaluation', 'consistency'):
        rows = load_pages(args.root / set_name, args.engines)
        outside = Counter(r['type'] for r in rows if r['type'] is not None and vocabulary is not None
                          and r['type'] not in vocabulary)
        usable = [r for r in rows if r['type'] is not None and (vocabulary is None or r['type'] in vocabulary)]
        section = {'pages': len(rows), 'without_type': sum(r['type'] is None for r in rows),
                   'outside_vocabulary': dict(outside), 'type': {}, 'side': {}, 'boundary': {}}
        groups = {'all': usable, **{g: [r for r in usable if r['group'] == g]
                                    for g in sorted({r['group'] for r in usable})},
                  'curated': [r for r in usable if r['curated']]}
        for name, group in groups.items():
            if not group:
                continue
            entry = {engine: classification([r['type'] for r in group], [r[engine]['type'] for r in group])
                     for engine in args.engines}
            right = [(r[first]['type'] == r['type'], r[second]['type'] == r['type']) for r in group]
            gained = sum(b and not a for a, b in right)
            lost = sum(a and not b for a, b in right)
            entry['second_right_first_wrong'], entry['first_right_second_wrong'] = gained, lost
            entry['mcnemar_p'] = float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0
            section['type'][name] = entry
        labels = sorted({r['type'] for r in usable})
        truth = [r['type'] for r in usable]
        table = per_class(truth, {e: [r[e]['type'] for r in usable] for e in args.engines}, labels)
        for label in labels:
            members = [r for r in usable if r['type'] == label]
            table[label] = {'fixed': sum(r[second]['type'] == label and r[first]['type'] != label for r in members),
                            'broken': sum(r[first]['type'] == label and r[second]['type'] != label for r in members)}
        section['per_type'] = {'labels': labels, 'table': table,
                               'confusions': {e: Counter((r['type'], r[e]['type']) for r in usable
                                                         if r[e]['type'] != r['type']).most_common(10)
                                              for e in args.engines}}
        sided = [r for r in rows if r['side'] is not None]
        side_labels = ['left', 'right', 'singlePage']
        section['per_side'] = {'labels': side_labels, 'table': per_class(
            [r['side'] for r in sided], {e: [r[e]['side'] or 'none' for r in sided] for e in args.engines},
            side_labels), 'confusion': {e: dict(Counter(f'{r["side"]} → {r[e]["side"] or "none"}' for r in sided))
                                        for e in args.engines}}
        for r in rows:
            r['boundary_group'] = boundary_group(r)
        section['side'] = {'n': len(sided), **{engine: (float(np.mean([r[engine]['side'] == r['side'] for r in sided]))
                                                       if sided and any(r[engine]['side'] for r in sided) else None)
                                               for engine in args.engines},
                           'reference': dict(Counter(r['side'] for r in sided))}
        for name, group in {'all': rows, **{g: [r for r in rows if r['boundary_group'] == g]
                                            for g in sorted({r['boundary_group'] for r in rows})}}.items():
            inside = [r for r in group if r['position'] > 0]
            if not inside:
                continue
            section['boundary'][name] = {
                engine: (boundary_scores([r['start'] for r in inside],
                                         [bool(r[engine]['start']) for r in inside])
                         if all(r[engine]['start'] is not None for r in inside) else None)
                for engine in args.engines}
        section['batches_by_group'] = dict(Counter(boundary_group(r) for r in rows if r['position'] == 0))
        report['sets'][set_name] = section
        csv_rows += [{'set': set_name, 'group': r['group'], 'batch': r['batch'], 'page': r['page'],
                      'position': r['position'], 'curated': r['curated'], 'truth_type': r['type'],
                      'truth_side': r['side'], 'truth_start': r['start'],
                      **{f'{e}_{k}': r[e][k] for e in args.engines for k in ('type', 'side', 'start')}}
                     for r in rows]
    (args.output / 'engine-comparison.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    markdown = render_markdown(report, args.engines)
    if args.conditions is not None:
        from metakat_checks.context_report import render_context_section
        markdown += '\n' + render_context_section(args.conditions)
    (args.output / 'engine-comparison.md').write_text(markdown)
    with (args.output / 'engine-comparison-pages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    for set_name, section in report['sets'].items():
        print(f'\n== {set_name}: {section["pages"]} pages, {section["without_type"]} without a reference type, '
              f'outside the vocabulary: {section["outside_vocabulary"]}')
        print(f'{"page type":<14}{"n":>6} | {first:>26} | {second:>26} | gained/lost  p')
        for name, e in section['type'].items():
            cell = lambda m: f'acc {m["accuracy"]:.3f} wF1 {m["weighted_f1"]:.3f}'
            print(f'{name:<14}{e[first]["n"]:>6} | {cell(e[first]):>26} | {cell(e[second]):>26} | '
                  f'{e["second_right_first_wrong"]:>4}/{e["first_right_second_wrong"]:<4} {e["mcnemar_p"]:.3f}')
        s = section['side']
        print(f'side (DB placement, {s["n"]} pages {s["reference"]}): ' +
              ', '.join(f'{e} {"none predicted" if s[e] is None else format(s[e], ".3f")}' for e in args.engines))
        for name, b in section['boundary'].items():
            cells = []
            for engine in args.engines:
                m = b[engine]
                cells.append('-' if m is None else f'P {m["precision"]:.3f} R {m["recall"]:.3f} F1 {m["f1"]:.3f} '
                             f'({m["hits"]}/{m["true_starts"]} starts, {m["predicted_starts"]} predicted)')
            print(f'starts {name:<12} {first}: {cells[0]} | {second}: {cells[1]}')


if __name__ == '__main__':
    main()
