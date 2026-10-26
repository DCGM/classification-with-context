from unittest.mock import Mock

import numpy as np
import pytest

from page_sequence.evaluation import Evaluator, classification_metrics, front_back_metrics
from page_sequence.schema import PAGE_TYPES, Page, LABEL2ID
from page_sequence.tracking import Tracking


def test_weighted_scores_and_directional_confusion():
    cm = np.zeros((len(PAGE_TYPES), len(PAGE_TYPES)), dtype=int)
    f, b, normal = (LABEL2ID[x] for x in ('FrontCover', 'BackCover', 'NormalPage'))
    cm[f, f], cm[f, b], cm[b, f], cm[normal, b] = 3, 1, 2, 4
    scores = classification_metrics(cm, PAGE_TYPES)
    assert scores['weighted_recall'] == scores['accuracy'] == .3
    assert scores['weighted_precision'] == pytest.approx(.24)
    reversal = front_back_metrics(cm)
    assert reversal['front_as_back'] == {'errors': 1, 'support': 4, 'rate': .25}
    assert reversal['back_as_front']['rate'] == 1
    assert reversal['combined']['rate'] == .5
    assert front_back_metrics(np.zeros_like(cm))['combined']['rate'] is None


def test_position_diagnostic_and_censoring():
    pages = [Page('a', 'd', str(i), i, annotated_type='NormalPage') for i in range(8)]
    result = {'type': np.array([LABEL2ID['BackCover']] + [LABEL2ID['NormalPage']]*6 + [LABEL2ID['FrontCover']]),
              'placement': np.zeros(8, int), 'starts_document_probs': np.array([np.nan]+[0.]*7),
              'pair_mask': np.array([False]+[True]*7), 'processed_pages': 8}
    e = Evaluator()
    e.add('complete', pages, result, {})
    r = e.report()
    assert r['front_back_position']['documents'] == 1
    assert r['front_back_position']['front_at_end']['rate'] == 1
    assert r['front_back_position']['back_at_start']['rate'] == 1
    # Wrong-end predictions on NormalPage are positional flags, not semantic reversals.
    assert r['front_back_confusion']['curated_type']['combined']['support'] == 0
    e = Evaluator()
    e.add('partial', pages, result, {}, censored_left=True)
    assert e.report()['front_back_position']['documents'] == 0


def test_clearml_receives_per_class_and_matrices():
    cm = np.zeros((len(PAGE_TYPES), len(PAGE_TYPES)), int)
    cm[0, 0] = 1
    report = Evaluator().report()
    report['curated_type'] = classification_metrics(cm, PAGE_TYPES)
    tracking = Tracking({},{})
    tracking.task, tracking.logger = Mock(), Mock()
    tracking.report({'step': 10, 'stage': 'pretrain', 'loss': {}, 'valid_weight': {},
                     'gradient_norm': 1, 'io_seconds': 1, 'pages_per_second': 1,
                     'coverage_epochs': 1, 'dev': report})
    calls = [c.args for c in tracking.logger.report_scalar.call_args_list]
    assert ('dev/curated_type/precision', 'Abstract', 1., 10) in calls
    assert ('dev/placement_db_agreement/support', 'single', 0, 10) in calls
    assert tracking.logger.report_confusion_matrix.call_count == 3
    matrix = tracking.logger.report_confusion_matrix.call_args_list[-1].kwargs
    assert matrix['xlabels'] == ['left', 'right', 'single']
    assert matrix['matrix'].shape == (3, 3)


def test_side_phase_reversal_versus_isolated_errors():
    from page_sequence.evaluation import side_sequence_counts, side_sequence_report
    pages = [Page('a', 'd', str(i), i) for i in range(8)]
    truth = np.array([0, 1]*4)
    reversed_report = side_sequence_report(side_sequence_counts(pages, truth, 1-truth))
    assert reversed_report['block_accuracy'] == 0
    assert reversed_report['oracle_phase_corrected_accuracy'] == 1
    assert reversed_report['perfectly_reversed_blocks'] == 1
    assert reversed_report['longest_reversed_run'] == 8
    assert reversed_report['swap_error_fraction_in_runs_ge4'] == 1
    isolated = truth.copy()
    isolated[[1, 5]] = 1-isolated[[1, 5]]
    report = side_sequence_report(side_sequence_counts(pages, truth, isolated))
    assert report['swap_error_fraction_in_runs_ge4'] == 0
    assert report['reversed_blocks'] == 0
    assert report['block_accuracy'] == report['oracle_phase_corrected_accuracy'] == .75
    # Predicting single remains wrong even after the diagnostic oracle swap.
    pred = 1-truth
    pred[3] = 2
    report = side_sequence_report(side_sequence_counts(pages, truth, pred))
    assert report['oracle_phase_corrected_accuracy'] == 7/8
    assert report['longest_reversed_run'] == 4


@pytest.mark.parametrize('break_kind', ['document', 'gap', 'missing', 'single', 'repeat'])
def test_side_runs_do_not_bridge_invalid_transitions(break_kind):
    from page_sequence.evaluation import side_sequence_counts, side_sequence_report
    pages = [Page('a', 'd', str(i), i) for i in range(8)]
    truth = np.array([0, 1]*4)
    if break_kind == 'document':
        pages[4:] = [Page('a', 'other', str(i), i) for i in range(4,8)]
    elif break_kind == 'gap':
        pages[4:] = [Page('a', 'd', str(i), i+1) for i in range(4,8)]
    elif break_kind in ('missing', 'single'):
        truth[4] = -100 if break_kind == 'missing' else 2
    else:
        truth[4:] = 1-truth[4:]
    pred = np.where((truth >= 0) & (truth < 2), 1-truth, 2)
    report = side_sequence_report(side_sequence_counts(pages, truth, pred))
    assert report['longest_reversed_run'] == 4
    assert report['alternating_blocks_ge4'] == (1 if break_kind in ('missing', 'single') else 2)
