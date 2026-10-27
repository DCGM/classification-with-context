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


def test_metadata_selected_per_complete_scan_and_restored_on_resume(tmp_path):
    from dataclasses import replace
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = BoundaryFeatureDataset(WindowDataset(index, features))
    settings = {'border_tokens': False, 'metadata_probability': .5}
    sampler = ScanBatchSampler(index, 6, scan_batches=settings)
    docs = index.documents('fit')[:3]
    decisions = []
    for _ in range(80):
        tiles = sampler._tiles(docs, 'test')
        decisions.append(tiles[0].document_metadata)
        assert len({w.document_metadata for w in tiles}) == 1
        for w in tiles:
            e = dataset[w]
            bits = e['features']['boundaries']
            assert (bits[:, 2] == w.document_metadata).all()
            if w.document_metadata:
                np.testing.assert_array_equal(bits[:, 3], e['page_start'])
                np.testing.assert_array_equal(bits[:, 4], e['page_end'])
            else:
                assert not bits[:, 2:].any()
            # Supplying metadata does not alter any target or the scan flags.
            other = dataset[replace(w, document_metadata=not w.document_metadata)]
            for name in ('type', 'gold', 'starts', 'page_start', 'page_end', 'pair_mask'):
                np.testing.assert_array_equal(e[name], other[name])
            np.testing.assert_array_equal(bits[:, :2], other['features']['boundaries'][:, :2])
    assert 20 < sum(decisions) < 60
    for _ in range(8): sampler.draw()
    restored = ScanBatchSampler(index, 6, scan_batches=settings)
    restored.load_state_dict(sampler.state_dict())
    assert [sampler.draw() for _ in range(30)] == [restored.draw() for _ in range(30)]


def test_inference_supplied_metadata_is_complete_and_preserved_across_tiles(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = BoundaryFeatureDataset(WindowDataset(index, features))
    docs = index.documents('fit')[:2]
    rows = tuple(r for d in docs for r in range(d.first, d.first+d.length))
    starts = np.array([any(r == d.first for d in docs) for r in rows], np.float32)
    ends = np.array([any(r == d.first+d.length-1 for d in docs) for r in rows], np.float32)
    class Model(torch.nn.Module):
        num_types = 39
        boundary_features = True
        def forward(self, f, mask):
            b,n = mask.shape
            assert (f['boundaries'][..., 2][mask] == 1).all()
            # Emit supplied starts as pair logits to check overlap alignment.
            pair = mask.clone(); pair[:, 0] = False
            return {'type_logits': torch.zeros(b,n,39), 'placement_logits': torch.zeros(b,n,3),
                    'starts_document_logits': (f['boundaries'][..., 3] * 2-1)*10,
                    'pair_mask': pair}
    result = predict_stream(Model(), dataset, rows, InferenceConfig(window_size=6,stride=3),
                            document_starts=starts, document_ends=ends)
    np.testing.assert_array_equal(result['starts'][1:], starts[1:].astype(bool))
    with pytest.raises(ValueError):
        predict_stream(Model(), dataset, rows, document_starts=starts)
    with pytest.raises(ValueError):
        predict_stream(Model(), dataset, rows, document_starts=starts[:-1],document_ends=ends)
