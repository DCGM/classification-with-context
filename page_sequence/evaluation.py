"""Score merged full streams once; type annotation and DB agreement stay distinct."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from .schema import IGNORE, PAGE_TYPES, PLACEMENTS, join_stratum

# Deliberately explicit: ambiguous Cover/Jacket/FlyLeaf and movable content are excluded.
FRONT_TYPES = ('FrontCover', 'FrontEndPaper', 'FrontEndSheet', 'FrontJacket', 'Frontispiece', 'TitlePage')
BACK_TYPES = ('BackCover', 'BackEndPaper', 'BackEndSheet')


def front_back_metrics(cm):
    front, back = ([PAGE_TYPES.index(name) for name in names] for names in (FRONT_TYPES, BACK_TYPES))
    result = {}
    for name, source, opposite in (('front_as_back', front, back), ('back_as_front', back, front)):
        n = int(cm[source].sum())
        errors = int(cm[np.ix_(source, opposite)].sum())
        result[name] = {'errors': errors, 'support': n, 'rate': errors / n if n else None}
    errors = sum(x['errors'] for x in result.values())
    n = sum(x['support'] for x in result.values())
    result['combined'] = {'errors': errors, 'support': n, 'rate': errors / n if n else None}
    return result


def classification_metrics(cm, names):
    support, predicted = cm.sum(1), cm.sum(0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    n = int(support.sum())
    present = (support + predicted) > 0
    return {'n': n, 'accuracy': float(tp.sum() / n) if n else None,
            'weighted_precision': float((precision * support).sum() / n) if n else None,
            'weighted_recall': float((recall * support).sum() / n) if n else None,
            'weighted_f1': float((f1 * support).sum() / n) if n else None,
            'macro_f1': float(f1[present].mean()) if present.any() else None,
            'per_class': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                'f1': float(f1[i]), 'support': int(support[i])}
                          for i, name in enumerate(names)}, 'confusion': cm.tolist()}


def binary_metrics(y, probabilities, threshold):
    y, probabilities = np.asarray(y, bool), np.asarray(probabilities, float)
    pred = probabilities >= threshold
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    neg = int((~y).sum())
    return {'n': len(y), 'positive': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'average_precision': float(average_precision_score(y, probabilities)) if y.any() else None,
            'false_splits_per_1000_negatives': 1000 * fp / neg if neg else None,
            'missed_boundary_rate': fn / int(y.sum()) if y.any() else None}


def spans(starts, censored_left=False, censored_right=False):
    n = len(starts)
    cuts = [0] + (np.flatnonzero(starts[1:]) + 1).tolist() + [n]
    result = list(zip(cuts, cuts[1:]))
    return {p for p in result if not (censored_left and p[0] == 0) and not (censored_right and p[1] == n)}


def side_sequence_counts(pages, truth, prediction):
    """Phase diagnostics on observed, consecutive, alternating left/right blocks."""
    counts = defaultdict(int)
    counts['swap_errors'] = int(np.sum((truth < 2) & (truth >= 0) &
                                      (prediction < 2) & (prediction >= 0) & (truth != prediction)))
    block_start = 0
    run = 0

    def finish_run():
        if run >= 4:
            counts['reversed_runs_ge4'] += 1
            counts['swap_errors_in_runs_ge4'] += run
        counts['longest_reversed_run'] = max(counts['longest_reversed_run'], run)

    def finish_block(end):
        if end - block_start < 4:
            return
        y, pred = truth[block_start:end], prediction[block_start:end]
        correct, flipped = int(np.sum(y == pred)), int(np.sum((1 - y) == pred))
        counts['alternating_blocks_ge4'] += 1
        counts['block_pages'] += len(y)
        counts['correct_pages'] += correct
        counts['oracle_correct_pages'] += max(correct, flipped)
        counts['reversed_blocks'] += int(flipped > correct)
        counts['perfectly_reversed_blocks'] += int(flipped == len(y))

    for i in range(len(pages)):
        continuation = (i > 0 and 0 <= truth[i - 1] < 2 and 0 <= truth[i] < 2
                        and truth[i] != truth[i - 1]
                        and pages[i].doc_key == pages[i - 1].doc_key
                        and pages[i].order == pages[i - 1].order + 1)
        if not continuation:
            finish_block(i)
            block_start = i
            finish_run()
            run = 0
        swapped = 0 <= truth[i] < 2 and 0 <= prediction[i] < 2 and truth[i] != prediction[i]
        if swapped:
            run += 1
        else:
            finish_run()
            run = 0
        if not 0 <= truth[i] < 2:
            block_start = i + 1
    finish_block(len(pages))
    finish_run()
    return counts


def side_sequence_report(counts):
    keys = ('swap_errors', 'reversed_runs_ge4', 'swap_errors_in_runs_ge4', 'longest_reversed_run',
            'alternating_blocks_ge4', 'block_pages', 'correct_pages', 'oracle_correct_pages',
            'reversed_blocks', 'perfectly_reversed_blocks')
    result = {key: counts.get(key, 0) for key in keys}
    for name, numerator, denominator in (
            ('swap_error_fraction_in_runs_ge4', 'swap_errors_in_runs_ge4', 'swap_errors'),
            ('block_accuracy', 'correct_pages', 'block_pages'),
            ('oracle_phase_corrected_accuracy', 'oracle_correct_pages', 'block_pages'),
            ('reversed_block_fraction', 'reversed_blocks', 'alternating_blocks_ge4')):
        result[name] = result[numerator] / result[denominator] if result[denominator] else None
    return result


class Evaluator:
    def __init__(self, threshold=.5):
        self.threshold = threshold
        self.confusions = {name: np.zeros((n, n), np.int64) for name, n in (
            ('curated_type', len(PAGE_TYPES)), ('weak_type_agreement', len(PAGE_TYPES)),
            ('placement_db_agreement', len(PLACEMENTS)))}
        self.boundaries = defaultdict(lambda: [[], []])
        self.streams = []
        self.span_tp = self.span_pred = self.span_true = 0
        self.position_counts = defaultdict(int)
        self.side_counts = defaultdict(int)

    def add(self, stream_id, pages, result, documents, censored_left=False, censored_right=False):
        if len(pages) != len(result['type']):
            raise ValueError('prediction alignment mismatch')
        labels, gold, side = (np.asarray(x) for x in zip(*(p.target() for p in pages)))
        side_counts = side_sequence_counts(pages, side, np.asarray(result['placement']))
        for name, value in side_counts.items():
            if name == 'longest_reversed_run':
                self.side_counts[name] = max(self.side_counts[name], value)
            else:
                self.side_counts[name] += value
        for name, y, pred, mask in (
            ('curated_type', labels, result['type'], (labels != IGNORE) & gold),
            ('weak_type_agreement', labels, result['type'], (labels != IGNORE) & ~gold),
            ('placement_db_agreement', side, result['placement'], side != IGNORE)):
            if not mask.any():
                continue
            self.confusions[name] += confusion_matrix(y[mask], pred[mask], labels=np.arange(len(self.confusions[name])))
        y = np.array([False] + [a.doc_key != b.doc_key for a, b in zip(pages, pages[1:])])
        valid = np.asarray(result['pair_mask'], bool)
        if valid[0] or not valid[1:].all():
            raise ValueError('evaluation requires all internal pairs exactly once')
        p = result['starts_document_probs']
        if not np.isfinite(p[valid]).all():
            raise ValueError('nonfinite merged boundary probabilities')
        for i in range(1, len(pages)):
            stratum = (join_stratum(documents[pages[i - 1].doc_key], documents[pages[i].doc_key])
                       if y[i] else 'within_document')
            for group in ('all', stratum):
                self.boundaries[group][0].append(y[i])
                self.boundaries[group][1].append(float(p[i]))
        predicted = p >= self.threshold
        predicted[0] = False
        gt_spans, pred_spans = spans(y, censored_left, censored_right), spans(predicted, censored_left, censored_right)
        # True document positions, never the model's possibly incorrect segmentation.
        # Rank quartiles avoid assuming page indices are gap-free.
        for start, end in gt_spans:
            if end - start < 4:
                continue
            self.position_counts['documents'] += 1
            for rank, label in enumerate(result['type'][start:end]):
                page_type = PAGE_TYPES[int(label)]
                position = (rank + .5) / (end - start)
                if page_type in FRONT_TYPES:
                    self.position_counts['front_predictions'] += 1
                    self.position_counts['front_at_end'] += int(position >= .75)
                elif page_type in BACK_TYPES:
                    self.position_counts['back_predictions'] += 1
                    self.position_counts['back_at_start'] += int(position < .25)
        self.span_tp += len(gt_spans & pred_spans)
        self.span_true += len(gt_spans)
        self.span_pred += len(pred_spans)
        self.streams.append({'id': stream_id, 'pages': len(pages),
                             'processed_pages': result['processed_pages'],
                             'exact_segmentation': bool(np.array_equal(y[1:], predicted[1:])),
                             'side_sequence': side_sequence_report(side_counts),
                             'starts': binary_metrics(y[1:], p[1:], self.threshold)})

    def report(self):
        return {name: classification_metrics(cm, PLACEMENTS if name.startswith('placement') else PAGE_TYPES)
                for name, cm in self.confusions.items()} | {
            'starts': {group: binary_metrics(*data, self.threshold) for group, data in self.boundaries.items()
                       if group in ('all', 'within_document')},
            # A join stratum contains only true starts. Reporting precision/F1 on it would be
            # misleading: false splits belong to non-boundary pairs, not to positive joins.
            'join_detection': {group: {
                k: v for k, v in binary_metrics(*data, self.threshold).items()
                if k in ('n', 'tp', 'fn', 'recall', 'missed_boundary_rate')}
                for group, data in self.boundaries.items() if group not in ('all', 'within_document')},
            'front_back_confusion': {name: front_back_metrics(self.confusions[name])
                                    for name in ('curated_type', 'weak_type_agreement')},
            'side_sequence_db_agreement': side_sequence_report(self.side_counts),
            'front_back_position': {
                'documents': self.position_counts['documents'],
                **{name: {'count': self.position_counts[name], 'predictions': self.position_counts[denom],
                          'rate': (self.position_counts[name] / self.position_counts[denom]
                                   if self.position_counts[denom] else None)}
                   for name, denom in (('front_at_end', 'front_predictions'),
                                       ('back_at_start', 'back_predictions'))}},
            'exact_spans': {'tp': self.span_tp, 'predicted': self.span_pred, 'true': self.span_true,
                            'precision': self.span_tp / self.span_pred if self.span_pred else None,
                            'recall': self.span_tp / self.span_true if self.span_true else None},
            'exact_stream_fraction': (sum(s['exact_segmentation'] for s in self.streams) / len(self.streams)
                                      if self.streams else None), 'streams': self.streams}
