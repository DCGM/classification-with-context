import numpy as np

from page_sequence.boundary_ambiguity import counts, metrics, page_group


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
