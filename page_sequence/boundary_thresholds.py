"""Exact empirical boundary operating points, with tied scores kept together.

Threshold selection on this same development panel is exploratory, not a new
held-out performance estimate. No model, checkpoint or deployment config changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def threshold_curve(truth, probabilities):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities, np.float64)
    if (truth.ndim != 1 or probabilities.shape != truth.shape or not len(truth)
            or not np.isin(truth, [0, 1]).all() or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('nonempty binary labels and finite probabilities in [0,1] required')
    truth = truth.astype(bool)
    order = np.argsort(-probabilities, kind='stable')
    p, y = probabilities[order], truth[order]
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    positives = int(truth.sum())
    tp = np.r_[0, np.cumsum(y, dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    fn = positives - tp
    thresholds = np.r_[np.nextafter(p[0], np.inf), p[ends]]
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp), float),
                   where=(2 * tp + fp + fn) > 0)
    return dict(threshold=thresholds, tp=tp, fp=fp, fn=fn, f1=f1,
                n=len(truth), positive=positives, negative=len(truth) - positives)


def point(curve, i):
    tp, fp, fn = (int(curve[k][i]) for k in ('tp', 'fp', 'fn'))
    threshold = float(curve['threshold'][i])
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0., f1=float(curve['f1'][i]),
                false_splits_per_1000_negatives=1000 * fp / curve['negative'] if curve['negative'] else None,
                threshold_in_unit_interval=0 <= threshold <= 1)


def point_at(curve, threshold):
    # Scores are included with >=, matching normal inference. The first curve
    # entry represents predicting no boundaries, even if scores saturate at 1.
    included = np.flatnonzero(curve['threshold'][1:] >= threshold)
    result = point(curve, int(included[-1] + 1) if len(included) else 0)
    result['threshold'] = threshold
    result['threshold_in_unit_interval'] = 0 <= threshold <= 1
    return result


def summarize(truth, probabilities):
    curve = threshold_curve(truth, probabilities)
    default = point_at(curve, .5)
    selected = {'default': default, 'best_f1': point(curve, int(curve['f1'].argmax()))}
    for name, budget in [('half_default_fn', default['fn'] // 2),
                         ('tenth_default_fn', default['fn'] // 10), ('zero_fn', 0),
                         ('recall_95', int(np.floor(curve['positive'] * .05))),
                         ('recall_99', int(np.floor(curve['positive'] * .01))),
                         ('recall_999', int(np.floor(curve['positive'] * .001)))]:
        # Highest threshold meeting the missed-boundary budget: minimum FP.
        selected[name] = point(curve, int(np.flatnonzero(curve['fn'] <= budget)[0]))
    for name, budget in [('half_default_fp', default['fp'] // 2),
                         ('tenth_default_fp', default['fp'] // 10), ('zero_fp', 0)]:
        eligible = curve['fp'] <= budget
        best_tp = curve['tp'][eligible].max()
        selected[name] = point(curve, int(np.flatnonzero(eligible & (curve['tp'] == best_tp))[0]))
    return dict(n=curve['n'], positives=curve['positive'], negatives=curve['negative'],
                operating_points=selected,
                fixed_thresholds=[point_at(curve, t) for t in
                                  [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]])


def analyze(root):
    reports = {}
    for name, runtime, store in json.loads((root / 'jobs.json').read_text()):
        directory = root / name
        if not (directory / 'complete.json').exists():
            raise ValueError(f'collection incomplete or unverified: {name}')
        metadata = json.loads((directory / 'metadata.json').read_text())
        values = {key: [] for key in ('truth', 'unknown', 'supplied')}
        for path in sorted(directory.glob('*.npz')):
            with np.load(path) as data:
                for key in values:
                    values[key].append(data[key])
        values = {key: np.concatenate(parts) for key, parts in values.items()}
        reports[name] = dict(checkpoint_step=metadata['checkpoint_step'],
                             checkpoint_sha256=metadata['checkpoint_sha256'],
                             inference=metadata['inference'],
                             **{mode: summarize(values['truth'], values[mode])
                                for mode in ('unknown', 'supplied')})
    report = dict(split='dev', exploratory=True, decision='probability >= threshold',
                  scope='original 193 development streams; first position of each stream excluded',
                  threshold_updates_applied=False, models=reports)
    (root / 'threshold-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root)
