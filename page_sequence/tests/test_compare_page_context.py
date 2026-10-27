import numpy as np
import pytest
import torch

from page_sequence.compare_page_context import compare, page_panel, paired_scores
from page_sequence.data import make_example
from page_sequence.inference import InferenceConfig
from page_sequence.schema import Document, IGNORE, PAGE_TYPES, Page


class Index:
    def __init__(self):
        self.docs = [Document(i, 'lib', str(i), 'dev', '', '', str(i), first, length)
                     for i, (first, length) in enumerate(((1, 2), (3, 5)))]
        self.records = {row: Page('lib', d.doc, str(row), row - d.first,
                                  db_type='Abstract', annotated_type='Abstract' if row % 2 else '')
                        for d in self.docs for row in range(d.first, d.first + d.length)}

    def pages(self, rows):
        return [self.records[r] for r in rows]

    def documents_for_rows(self, rows):
        return [d for d in self.docs if any(d.first <= r < d.first + d.length for r in rows)]


def test_panel_preserves_order_and_rejects_missing_or_repeated_pages():
    index = Index()
    streams = [{'rows': [3, 4, 5, 6, 7]}, {'rows': [1, 2]}]
    rows, spans, labels, gold = page_panel(index, streams)
    assert rows == (3, 4, 5, 6, 7, 1, 2)
    assert spans == [(0, 5), (5, 7)]
    assert labels.tolist() == [0] * 7
    assert gold.tolist() == [True, False, True, False, True, True, False]
    for bad in ([1, 1], [1], [2, 1], [1, 2, 3, 5, 6, 7]):
        with pytest.raises(ValueError):
            page_panel(index, [{'rows': bad}])


def test_paired_counts_keep_weak_labels_separate():
    r = paired_scores(np.array([0, 1, 2, IGNORE]), np.array([True, True, False, False]),
                      np.array([0, 0, 2, 0]), np.array([1, 1, 0, 1]))
    assert r['all_pages_changed'] == 4
    assert r['curated_type']['continuous_fixes'] == 1
    assert r['curated_type']['continuous_regressions'] == 1
    assert r['weak_type_agreement']['continuous_regressions'] == 1
    assert r['weak_type_agreement']['n'] == 1


def test_compare_real_window_merging_same_pages_and_full_continuous_windows(tmp_path):
    index = Index()

    class Dataset:
        def __getitem__(self, window):
            return make_example(index.pages(window.rows),
                                lambda pages: {'image': np.asarray([[float(p.doc)] for p in pages], np.float32)},
                                window)

    class Model(torch.nn.Module):
        num_types = len(PAGE_TYPES)

        def forward(self, features, mask):
            b, n = mask.shape
            logits = torch.zeros(b, n, self.num_types)
            logits[:, :, 0] = 1
            logits[:, :, 1] = (features['image'].squeeze(-1) * mask).sum(1)[:, None] / mask.sum(1)[:, None] * 3
            pair = mask.clone()
            pair[:, 0] = False
            return dict(type_logits=logits, placement_logits=torch.zeros(b, n, 3),
                        starts_document_logits=torch.zeros(b, n), pair_mask=pair)

    dataset = Dataset()
    dataset.index = index
    output = tmp_path / 'comparison'
    report = compare(Model(), dataset, [{'rows': list(range(1, 8))}],
                     InferenceConfig(window_size=4, stride=2, batch_size=2), output)
    assert report['pages'] == 7 and report['documents'] == 2
    assert report['modes']['continuous']['full_windows'] == report['modes']['continuous']['windows'] == 3
    assert report['modes']['per_document']['full_windows'] == 2
    for mode in ('per_document', 'continuous'):
        probs = np.load(output / f'{mode}-probabilities.npy')
        assert probs.shape == (7, len(PAGE_TYPES))
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
        assert report['modes'][mode]['curated_type']['n'] == 4
        assert report['modes'][mode]['weak_type_agreement']['n'] == 3
    assert report['paired']['all_pages_changed'] > 0
