import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

from page_sequence.boundary_thresholds import high_threshold_analysis, point_at, summarize, threshold_curve


def test_tied_scores_cannot_be_split_to_remove_false_positives():
    result = summarize([1, 0, 1, 0, 1], [.9, .9, .7, .4, .2])['operating_points']
    assert (result['default']['tp'], result['default']['fp'], result['default']['fn']) == (2, 1, 1)
    assert result['best_f1']['threshold'] == .2
    assert (result['zero_fn']['fp'], result['zero_fn']['fn']) == (2, 0)
    assert (result['zero_fp']['fp'], result['zero_fp']['fn']) == (0, 3)
    assert result['zero_fp']['threshold'] > .9


def test_saturated_scores_expose_when_zero_fp_requires_no_predictions():
    p = summarize([0, 1], [1., 1.])['operating_points']['zero_fp']
    assert not p['threshold_in_unit_interval'] and p['threshold'] > 1
    assert p['fp'] == 0 and p['fn'] == 1


def test_curve_matches_sklearn_and_inclusive_threshold_decisions():
    rng = np.random.default_rng(17)
    truth = rng.random(1000) < .1
    probs = np.round(rng.random(1000), 2)
    curve = threshold_curve(truth, probs)
    precision, recall, _ = precision_recall_curve(truth, probs)
    expected = np.divide(2 * precision * recall, precision + recall,
                         out=np.zeros_like(precision), where=(precision + recall) > 0).max()
    assert curve['f1'].max() == pytest.approx(expected)
    for threshold in [0., .01, .5, .99, 1.]:
        point = point_at(curve, threshold)
        pred = probs >= threshold
        assert point['fp'] == int((~truth & pred).sum())
        assert point['fn'] == int((truth & ~pred).sum())


def test_higher_false_positive_cost_selects_conservative_operating_point():
    result = high_threshold_analysis(threshold_curve([1, 1, 0, 1], [.9, .8, .85, .4]))
    optimum = result['cost_optima_above_half']['2.0']
    assert optimum['threshold'] == .9
    assert (optimum['fp'], optimum['fn'], optimum['weighted_errors']) == (0, 2, 2.)
    # If every score is below 0.5, the allowed range still includes the default.
    result = high_threshold_analysis(threshold_curve([1], [.3]))
    assert result['cost_optima_above_half']['2.0']['threshold'] == .5
