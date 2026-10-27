import numpy as np
import pytest
import torch

from page_sequence.augmentation import modality_dropout, scattered_dropout
from page_sequence.data import WindowDataset, collate_windows
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.inference import InferenceConfig, predict_stream, tag_pages
from page_sequence.losses import task_losses
from page_sequence.models import BertBackbone, PageTagger, from_specification
from page_sequence.scan_borders import BoundaryFeatureDataset, encode_boundary_features, scan_windows
from page_sequence.scan_sampling import ScanBatchSampler
from page_sequence.schema import Window
from page_sequence.smoke import prepare_fixture


def test_five_bits_preserve_unknown_and_require_complete_external_metadata():
    np.testing.assert_array_equal(encode_boundary_features(1, True, True), [[1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(encode_boundary_features(3, True, True),
                                  [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    supplied = encode_boundary_features(3, True, True, [1, 0, 1], [0, 1, 1])
    np.testing.assert_array_equal(supplied[:, 2:], [[1, 1, 0], [1, 0, 1], [1, 1, 1]])
    for args in (([1], None), ([1], [0]), ([1, 0, 0], [0, 2, 1])):
        with pytest.raises(ValueError):
            encode_boundary_features(3, document_starts=args[0], document_ends=args[1])


@pytest.mark.parametrize('size', [256, 512])
def test_feature_tiling_keeps_every_page_slot(size):
    assert scan_windows(size, size, size // 2, False) == [(0, size, True, True)]
    tiles = scan_windows(3 * size + 1, size, size // 2, False)
    assert all(hi - lo == size for lo, hi, _, _ in tiles)
    assert sum(begin for _, _, begin, _ in tiles) == sum(end for _, _, _, end in tiles) == 1
    assert set(i for lo, hi, _, _ in tiles for i in range(lo + 1, hi)) == set(range(1, 3 * size + 1))


def test_sampler_features_no_target_leak_dropout_gradients_and_inference(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = BoundaryFeatureDataset(WindowDataset(index, features))
    sampler = ScanBatchSampler(index, 12, variable_lengths=True, scan_batches={'border_tokens': False})
    for _ in range(100):
        w = sampler.draw()
        values = dataset[w]['features']['boundaries']
        assert not values[:, 2:].any()
        assert values[:, 0].sum() == w.scan_start
        assert values[:, 1].sum() == w.scan_end
        assert len(w.rows) == 12 or (w.scan_start and w.scan_end)
    clone = ScanBatchSampler(index, 12, variable_lengths=True, scan_batches={'border_tokens': False})
    clone.load_state_dict(sampler.state_dict())
    assert [sampler.draw() for _ in range(20)] == [clone.draw() for _ in range(20)]
    with pytest.raises(ValueError, match='scan batch policy'):
        ScanBatchSampler(index, 12, variable_lengths=True).load_state_dict(sampler.state_dict())
    d = index.documents('fit')[0]
    example = dataset[Window(tuple(range(d.first, d.first + d.length)), scan_start=True, scan_end=True)]
    batch = collate_windows([example])
    original = batch['features']['boundaries'].clone()
    masked = modality_dropout(batch['features'], all_probability=0, single_probability=1)
    masked = scattered_dropout(masked, modality_probability=1, page_probability=1)
    torch.testing.assert_close(masked['boundaries'], original)
    backbone = BertBackbone.create(**{k: v for k, v in config['backbone'].items() if k != 'kind'})
    model = PageTagger(backbone, dataset.model_dims, 'separate_mlp', auxiliary_edges=True).eval()
    output = model(batch['features'], batch['attention_mask'])
    loss, _ = task_losses(output, batch)
    loss['total'].backward()
    assert model.fusion.branches['boundaries'][0].weight.grad.abs().sum() > 0
    # Even the all-unknown interior case must retain a trainable branch.
    unknown = {k: v.clone() for k, v in batch['features'].items()}
    unknown['boundaries'].zero_()
    model.zero_grad()
    model(unknown, batch['attention_mask'])['type_logits'].sum().backward()
    assert model.fusion.branches['boundaries'][0].bias.grad.abs().sum() > 0
    restored = from_specification(model.specification())
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.boundary_features and not restored.scan_borders
    assert not hasattr(restored, 'scan_border_embeddings')
    rows = tuple(range(d.first, d.first + d.length))
    result = predict_stream(restored, dataset, rows, InferenceConfig(window_size=12, stride=6))
    # Public inference without any catalogue also creates the five bits correctly.
    public = tag_pages(restored, features, index.pages(rows), InferenceConfig(window_size=12, stride=6))
    np.testing.assert_allclose(result['type_probs'], public['type_probs'], atol=1e-7)
