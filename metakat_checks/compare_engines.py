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


def boundary_scores(truth, prediction):
    truth, prediction = np.array(truth), np.array(prediction)
    tp = int((truth & prediction).sum())
    precision = tp / max(int(prediction.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    return {'pages': len(truth), 'true_starts': int(truth.sum()), 'predicted_starts': int(prediction.sum()),
            'hits': tp, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/kolosus/data/metakat_worker/test_batches/page_types'))
    parser.add_argument('--engines', nargs=2, default=['chapter-v1.6.0', 'chapter-v1.7.0'])
    parser.add_argument('--engine-configs', nargs=2, type=Path, default=[
        Path('/mnt/kolosus/data/metakat_worker/src_engines/chapter.v1.6.0.engine.json'),
        Path('/mnt/kolosus/data/metakat_worker/src_engines/chapter.v1.7.0.engine.json')])
    parser.add_argument('--output', type=Path, required=True)
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
        sided = [r for r in rows if r['side'] is not None]
        section['side'] = {'n': len(sided), **{engine: (float(np.mean([r[engine]['side'] == r['side'] for r in sided]))
                                                       if sided and any(r[engine]['side'] for r in sided) else None)
                                               for engine in args.engines},
                           'reference': dict(Counter(r['side'] for r in sided))}
        for name, group in {'all': rows, **{g: [r for r in rows if r['group'] == g]
                                            for g in sorted({r['group'] for r in rows})}}.items():
            inside = [r for r in group if r['position'] > 0]
            if not inside:
                continue
            section['boundary'][name] = {
                engine: (boundary_scores([r['start'] for r in inside],
                                         [bool(r[engine]['start']) for r in inside])
                         if all(r[engine]['start'] is not None for r in inside) else None)
                for engine in args.engines}
        report['sets'][set_name] = section
        csv_rows += [{'set': set_name, 'group': r['group'], 'batch': r['batch'], 'page': r['page'],
                      'position': r['position'], 'curated': r['curated'], 'truth_type': r['type'],
                      'truth_side': r['side'], 'truth_start': r['start'],
                      **{f'{e}_{k}': r[e][k] for e in args.engines for k in ('type', 'side', 'start')}}
                     for r in rows]
    (args.output / 'engine-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
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
