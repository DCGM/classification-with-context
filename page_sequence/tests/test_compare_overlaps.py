import numpy as np
import pytest
import torch

from page_sequence.compare_overlaps import sweep
from page_sequence.data import make_example
from page_sequence.inference import InferenceConfig
from page_sequence.schema import PAGE_TYPES
from page_sequence.tests.test_compare_page_context import Index


@pytest.mark.parametrize('ramp_fraction', [None, .5])
@pytest.mark.parametrize('ignore_fraction', [0., .125])
@pytest.mark.parametrize('processing', ['continuous', 'per_document'])
@pytest.mark.parametrize('tail', ['anchored', 'regular'])
def test_overlap_sweep_preserves_panel_and_supplies_complete_metadata(tmp_path, processing, tail, ignore_fraction, ramp_fraction):
    index = Index()

    class Dataset:
        def __getitem__(self, window):
            return make_example(index.pages(window.rows),
                                lambda pages: {'image': np.ones((len(pages), 1), np.float32)}, window)

    class Model(torch.nn.Module):
        num_types = len(PAGE_TYPES)
        boundary_features = True

        def forward(self, features, mask):
            b, n = mask.shape
            assert features['boundaries'].shape == (b, n, 5)
            logits = torch.zeros(b, n, self.num_types)
            logits[:, :, 0] = 1
            pair = mask.clone()
            pair[:, 0] = False
            return dict(type_logits=logits, placement_logits=torch.zeros(b, n, 3),
                        starts_document_logits=torch.zeros(b, n), pair_mask=pair)

    dataset = Dataset()
    dataset.index = index
    report = sweep(Model(), dataset, [{'rows': list(range(1, 8))}],
                   InferenceConfig(window_size=4, stride=2, batch_size=2, tail=tail), [2, 1], tmp_path / 'sweep',
                   processing=processing, edge_ignore_fraction=ignore_fraction, edge_ramp_fraction=ramp_fraction)
    assert report['pages'] == 7 and report['documents'] == 2
    for mode in ('unknown', 'provided'):
        for overlap in (2, 1):
            entry = report['results'][mode][str(overlap)]
            assert entry['inference']['stride'] == 4 - overlap
            assert entry['inference']['edge_ignore'] == int(overlap * ignore_fraction)
            assert entry['inference']['edge_ramp'] == (32 if ramp_fraction is None else max(1, int(overlap * ramp_fraction)))
            assert entry['curated_type']['n'] == 4
            assert entry['starts']['n'] == (6 if processing == 'continuous' else 5)
            assert entry['starts']['positive'] == (1 if processing == 'continuous' else 0)
            assert entry['curated_long_documents']['n'] == 3
            assert entry['curated_type']['weighted_f1'] == 1
            assert entry['paired_against_baseline']['all_pages_changed'] == 0
        assert report['results'][mode]['1']['processed_pages'] <= report['results'][mode]['2']['processed_pages']
