import pickle

import numpy as np
import pytest
import torch

from page_sequence.augmentation import modality_dropout, scattered_dropout
from page_sequence.modalities import MODALITIES, ModalitySubsetDataset, enabled_modalities
from page_sequence.models import PageBackbone, PageTagger, from_specification

SUBSETS = [('stats',), ('text',), ('image',), ('stats', 'text'), ('stats', 'image'), ('text', 'image')]


class ExampleDataset:
    def __getitem__(self, window):
        return {'features': {**{k: np.ones((4, 5), np.float32) for k in MODALITIES},
                             'presence': np.ones((4, 3), np.float32),
                             'pagination': np.ones((4, 5), np.float32),
                             'boundaries': np.ones((4, 5), np.float32)}, 'rows': [1, 2, 3, 4]}


@pytest.mark.parametrize('subset', SUBSETS)
def test_permanent_dataset_mask_survives_workers_and_dropout_rescue(subset):
    dataset = pickle.loads(pickle.dumps(ModalitySubsetDataset(ExampleDataset(), subset)))
    result = dataset[None]['features']
    expected = [int(k in subset) for k in MODALITIES]
    np.testing.assert_array_equal(result['presence'], np.tile(expected, (4, 1)))
    features = {k: torch.from_numpy(v)[None].expand(100, -1, -1) for k, v in result.items()}
    dropped = modality_dropout(features, all_probability=0, two_probability=0, single_probability=1)
    dropped = scattered_dropout(dropped, modality_probability=.1, page_probability=.05)
    for i, name in enumerate(MODALITIES):
        if name not in subset:
            assert not dropped[name].any() and not dropped['presence'][..., i].any()
    np.testing.assert_array_equal(result['pagination'], 1)
    np.testing.assert_array_equal(result['boundaries'], 1)


@pytest.mark.parametrize('subset', SUBSETS)
def test_model_enforces_subset_even_on_unmasked_inference_and_roundtrip(subset):
    dims = dict(image=5, text=5, stats=5, pagination=5, boundaries=5)
    torch.manual_seed(17)
    model = PageTagger(PageBackbone(16), dims, 'separate_mlp', auxiliary_edges=True,
                       enabled_modalities=subset).eval()
    features = {k: torch.randn(2, 4, n) for k, n in dims.items()}
    features['presence'] = torch.ones(2, 4, 3)
    features['pagination'][..., 4] = 1
    attention = torch.ones(2, 4, dtype=torch.bool)
    original = model(features, attention)
    perturbed = {k: v.clone() for k, v in features.items()}
    for i, name in enumerate(MODALITIES):
        if name not in subset:
            perturbed[name] = torch.randn_like(perturbed[name]) * 100
            perturbed['presence'][..., i] = 0
    changed = model(perturbed, attention)
    for key in original:
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    restored = from_specification(model.specification()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.enabled_modalities == enabled_modalities(subset)
    for key, value in restored(features, attention).items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    original['type_logits'].sum().backward()
    for name in MODALITIES:
        if name not in subset:
            assert all(p.grad is None or not p.grad.any() for p in model.fusion.branches[name].parameters())


def test_invalid_subsets_are_rejected():
    for value in [[], ['stats', 'stats'], ['unknown'], 'stats']:
        with pytest.raises(ValueError): enabled_modalities(value)


def test_evaluation_modality_subsets():
    import pytest
    from page_sequence.evaluate_conditions import modality_subsets
    assert modality_subsets(['all'], ('image', 'stats')) == [('image', 'stats'), ('image',), ('stats',)]
    assert len(modality_subsets(['all'], ('image', 'text', 'stats'))) == 7
    assert modality_subsets(['stats+image'], ('image', 'text', 'stats')) == [('image', 'stats')]
    with pytest.raises(ValueError):
        modality_subsets(['text'], ('image', 'stats'))
