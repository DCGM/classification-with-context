"""Boundary metrics by the reference type of the preceding page, using saved scores.

The category is independent of model predictions and applies to every eligible
pair, including within-document pairs, so false-positive denominators are valid.
Unknown labels are reported separately. No source OCR or page text is read.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from .schema import PAGE_TYPES


AMBIGUOUS_END_TYPES = ('CalibrationTable', 'Cover', 'Jacket', 'FlyLeaf', 'Blank',
                       'Spine', 'Edge', 'FragmentsOfBookbinding')


def page_group(label, ambiguous_types=AMBIGUOUS_END_TYPES):
    if label not in PAGE_TYPES:
        return 'unknown'
    return 'ambiguous' if label in ambiguous_types else 'not_ambiguous'


def counts(truth, probabilities, threshold):
    truth = np.asarray(truth, bool)
    probabilities = np.asarray(probabilities)
    if truth.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError('aligned finite probabilities required')
    predicted = probabilities >= threshold
    return Counter(n=len(truth), positive=int(truth.sum()), tp=int((truth & predicted).sum()),
                   fp=int((~truth & predicted).sum()), fn=int((truth & ~predicted).sum()),
                   tn=int((~truth & ~predicted).sum()))


def metrics(c):
    c = {name: int(c.get(name, 0)) for name in ('n', 'positive', 'tp', 'fp', 'fn', 'tn')}
    tp, fp, fn = c['tp'], c['fp'], c['fn']
    negative = c['n'] - c['positive']
    return dict(c, negative=negative,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                missed_boundary_rate=fn / c['positive'] if c['positive'] else None,
                false_splits_per_1000_negatives=1000 * fp / negative if negative else None)


def nearest_boundary_offsets(truth, predicted):
    """Nearest detected boundary in sequence positions, within this one scan.

    Negative offsets mean an early cut. A prediction may cover more than one
    nearby truth: this is a recall-only proximity diagnostic, not relabelling FP.
    """
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    if truth.ndim != 1 or truth.shape != predicted.shape:
        raise ValueError('aligned one-dimensional boundary masks required')
    positions, detected = np.flatnonzero(truth), np.flatnonzero(predicted)
    if not len(detected):
        return positions, np.full(len(positions), np.inf)
    after = np.searchsorted(detected, positions)
    left = detected[np.maximum(after - 1, 0)] - positions
    right = detected[np.minimum(after, len(detected) - 1)] - positions
    # The clipped candidate is still a valid boundary when one side is absent.
    return positions, np.where(np.abs(left) <= np.abs(right), left, right)


def one_to_one_boundary_hits(truth, predicted, radius):
    """Maximum matched count with each truth/prediction used at most once.

    Retained as a sensitivity check for closely spaced document boundaries;
    it does not replace the requested any-nearby-prediction recall metric.
    """
    actual, detected = np.flatnonzero(truth), np.flatnonzero(predicted)
    i = j = hits = 0
    while i < len(actual) and j < len(detected):
        if detected[j] < actual[i] - radius:
            j += 1
        elif detected[j] > actual[i] + radius:
            i += 1
        else:
            hits += 1
            i += 1
            j += 1
    return hits


def recall_metrics(c):
    total, found = int(c.get('total', 0)), int(c.get('found', 0))
    return dict(total=total, found=found, missed=total - found,
                recall=found / total if total else None)


def reference_labels(db, rows):
    lookup = {}
    unique = list(dict.fromkeys(map(int, rows)))
    for start in range(0, len(unique), 512):
        chunk = unique[start:start + 512]
        for row, weak, gold in db.execute('SELECT id,db_type,annotated_type FROM pages WHERE id IN ('
                                          + ','.join('?' for _ in chunk) + ')', chunk):
            lookup[row] = (gold or weak, bool(gold))
    if len(lookup) != len(unique):
        raise ValueError('some scored pages are absent from catalogue')
    return (np.asarray([lookup[int(row)][0] for row in rows], object),
            np.asarray([lookup[int(row)][1] for row in rows], bool))


def analyze(root, catalogue, threshold=.5, max_tolerance=5):
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between zero and one')
    if not isinstance(max_tolerance, int) or max_tolerance < 0:
        raise ValueError('nonnegative integer tolerance required')
    jobs = json.loads((root / 'jobs.json').read_text())
    db = sqlite3.connect(Path(catalogue).resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-32768')
    metadata = {}
    for name, runtime, store in jobs:
        if not (root / name / 'complete.json').exists():
            raise ValueError('boundary score collection is not complete')
        metadata[name] = json.loads((root / name / 'metadata.json').read_text())
    totals = {name: {mode: defaultdict(Counter) for mode in ('unknown', 'supplied')}
              for name, _, _ in jobs}
    per_type = {name: {mode: defaultdict(Counter) for mode in ('unknown', 'supplied')}
                for name, _, _ in jobs}
    tolerant = {name: {mode: {str(k): defaultdict(Counter) for k in range(max_tolerance + 1)}
                       for mode in ('unknown', 'supplied')} for name, _, _ in jobs}
    tolerant_types = {name: {mode: {str(k): defaultdict(Counter) for k in range(max_tolerance + 1)}
                             for mode in ('unknown', 'supplied')} for name, _, _ in jobs}
    offset_counts = {name: {mode: defaultdict(Counter) for mode in ('unknown', 'supplied')}
                     for name, _, _ in jobs}
    unique_matches = {name: {mode: Counter() for mode in ('unknown', 'supplied')}
                       for name, _, _ in jobs}
    reference_sources = Counter()
    names = sorted((root / jobs[0][0]).glob('*.npz'))
    if len(names) != metadata[jobs[0][0]]['streams']:
        raise ValueError('incomplete stream set')
    for source in names:
        with np.load(source) as values:
            previous, rows, truth = values['previous_rows'], values['rows'], values['truth']
        labels, curated = reference_labels(db, previous)
        groups = np.asarray([page_group(label) for label in labels], object)
        reference_sources.update(pairs_curated=int(curated.sum()), pairs_db=int((~curated).sum()),
                                 boundaries_curated=int((curated & truth).sum()),
                                 boundaries_db=int((~curated & truth).sum()))
        for name, runtime, store in jobs:
            with np.load(root / name / source.name) as values:
                if not all(np.array_equal(values[key], expected) for key, expected in
                           [('previous_rows', previous), ('rows', rows), ('truth', truth)]):
                    raise ValueError('model evaluation pairs are not aligned')
                for mode in ('unknown', 'supplied'):
                    p = values[mode]
                    totals[name][mode]['all'].update(counts(truth, p, threshold))
                    positions, offsets = nearest_boundary_offsets(truth, p >= threshold)
                    distances = np.abs(offsets)
                    positive_groups, positive_labels = groups[positions], labels[positions]
                    for group in ('all', 'ambiguous', 'not_ambiguous', 'unknown'):
                        mask = (np.ones(len(positions), bool) if group == 'all' else positive_groups == group)
                        for k in range(max_tolerance + 1):
                            tolerant[name][mode][str(k)][group].update(
                                total=int(mask.sum()), found=int((mask & (distances <= k)).sum()))
                        for offset in offsets[mask]:
                            key = str(int(offset)) if abs(offset) <= max_tolerance else 'outside_tolerance_or_no_prediction'
                            offset_counts[name][mode][group][key] += 1
                    for label in set(positive_labels):
                        mask = positive_labels == label
                        for k in range(max_tolerance + 1):
                            tolerant_types[name][mode][str(k)][label or '(missing)'].update(
                                total=int(mask.sum()), found=int((mask & (distances <= k)).sum()))
                    for k in range(max_tolerance + 1):
                        unique_matches[name][mode][str(k)] += one_to_one_boundary_hits(truth, p >= threshold, k)
                    for group in ('ambiguous', 'not_ambiguous', 'unknown'):
                        mask = groups == group
                        totals[name][mode][group].update(counts(truth[mask], p[mask], threshold))
                    for label in set(labels):
                        mask = labels == label
                        per_type[name][mode][label or '(missing)'].update(counts(truth[mask], p[mask], threshold))
    db.close()
    result = dict(split='dev', threshold=threshold, ambiguous_preceding_types=AMBIGUOUS_END_TYPES,
                  grouping='reference type of preceding page for every eligible adjacent pair',
                  reference_policy='human annotation if present, otherwise DB label; unknown separate',
                  reference_sources=dict(reference_sources), models={},
                  tolerance_definition='recall-only: any detected boundary within +/- k observed scan positions; '
                                       'never across streams; original truth category retained; '
                                       'precision/FP remain exact; predictions may cover multiple nearby truths',
                  max_tolerance=max_tolerance)
    for name, runtime, store in jobs:
        report = {}
        for mode in ('unknown', 'supplied'):
            groups = {group: metrics(c) for group, c in totals[name][mode].items()}
            for field in ('n', 'positive', 'tp', 'fp', 'fn', 'tn'):
                assert sum(groups[g][field] for g in ('ambiguous', 'not_ambiguous', 'unknown')) == groups['all'][field]
            if threshold == .5:
                expected = metadata[name]['expected_' + mode]
                assert all(groups['all'][k] == expected[k] for k in ('n', 'positive', 'tp', 'fp', 'fn'))
            report[mode] = dict(groups=groups, preceding_page_types={
                label: dict(metrics(c), group=page_group(label))
                for label, c in sorted(per_type[name][mode].items())},
                recall_tolerance={radius: {g: recall_metrics(c) for g, c in values.items()}
                                  for radius, values in tolerant[name][mode].items()},
                type_recall_tolerance={radius: {label: recall_metrics(c) for label, c in values.items()}
                                       for radius, values in tolerant_types[name][mode].items()},
                nearest_boundary_offsets={g: dict(c) for g, c in offset_counts[name][mode].items()},
                one_to_one_found_sensitivity=dict(unique_matches[name][mode]))
            for group in groups:
                exact = report[mode]['recall_tolerance']['0'][group]
                assert exact['found'] == groups[group]['tp']
                assert exact['missed'] == groups[group]['fn']
                recovered = [report[mode]['recall_tolerance'][str(k)][group]['found']
                             for k in range(max_tolerance + 1)]
                assert recovered == sorted(recovered)
        result['models'][name] = dict(checkpoint_step=metadata[name]['checkpoint_step'], **report)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--catalogue', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--max-tolerance', type=int, default=5)
    args = parser.parse_args()
    report = analyze(args.root, args.catalogue, args.threshold, args.max_tolerance)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
