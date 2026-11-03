import numpy as np

from page_sequence.boundary_ambiguity import (counts, metrics, page_group,
                                             nearest_boundary_offsets, one_to_one_boundary_hits)


def test_directional_labels_and_missing_types_are_not_silently_ambiguous_or_clear():
    assert page_group('Cover') == 'ambiguous'
    assert page_group('CalibrationTable') == 'ambiguous'
    assert page_group('BackCover') == 'not_ambiguous'
    assert page_group('FrontCover') == 'not_ambiguous'
    assert page_group('') == 'unknown'
    assert page_group('unsupported') == 'unknown'


def test_partition_includes_within_document_pairs_for_false_positive_counts():
    truth = np.asarray([1, 1, 0, 0, 1, 0], bool)
    probabilities = np.asarray([.8, .2, .9, .1, .5, .7])
    labels = ['Cover', 'BackCover', 'Cover', 'NormalPage', '', '']
    groups = np.asarray([page_group(label) for label in labels])
    overall = counts(truth, probabilities, .5)
    parts = {g: counts(truth[groups == g], probabilities[groups == g], .5)
             for g in ('ambiguous', 'not_ambiguous', 'unknown')}
    for key in overall:
        assert sum(c[key] for c in parts.values()) == overall[key]
    assert parts['ambiguous']['fp'] == 1  # An internal Cover is not a true document end.
    assert metrics(parts['ambiguous'])['precision'] == .5
    assert metrics(parts['not_ambiguous'])['recall'] == 0
    assert parts['unknown']['tp'] == 1 and parts['unknown']['fp'] == 1


def test_tolerant_recall_recovers_early_and_late_predictions_without_changing_fp():
    truth = np.zeros(20, bool); truth[[2, 8, 15]] = True
    predicted = np.zeros(20, bool); predicted[[1, 10, 19]] = True
    positions, offsets = nearest_boundary_offsets(truth, predicted)
    np.testing.assert_array_equal(positions, [2, 8, 15])
    np.testing.assert_array_equal(offsets, [-1, 2, 4])
    assert [int((np.abs(offsets) <= k).sum()) for k in range(6)] == [0, 1, 2, 2, 3, 3]
    assert counts(truth, predicted.astype(float), .5)['fp'] == 3
    # Separate calls define separate scans: a cut at the previous scan's end
    # cannot recover a boundary at the beginning of this scan.
    _, absent = nearest_boundary_offsets([1, 0], [0, 0])
    assert np.isinf(absent).all()


def test_nearby_truths_can_share_a_prediction_only_in_recall_only_diagnostic():
    truth = np.asarray([0, 1, 0, 1, 0], bool)
    predicted = np.asarray([0, 0, 1, 0, 0], bool)
    _, offsets = nearest_boundary_offsets(truth, predicted)
    assert (np.abs(offsets) <= 1).sum() == 2
    assert one_to_one_boundary_hits(truth, predicted, 1) == 1
