import json

import numpy as np
import pytest
import torch

from page_sequence.augmentation import scattered_dropout
from page_sequence.data import WindowDataset, collate_windows
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.inference import InferenceConfig, predict_stream
from page_sequence.losses import task_losses
from page_sequence.models import BertBackbone, PageTagger, from_specification
from page_sequence.runner import load_checkpoint, train
from page_sequence.scan_borders import ScanBorderDataset, scan_windows
from page_sequence.scan_sampling import ScanBatchSampler
from page_sequence.schema import Window
from page_sequence.smoke import prepare_fixture


@pytest.mark.parametrize('size', [3, 12, 256, 512])
def test_tiling_uses_network_positions_and_covers_every_page_and_pair(size):
    for n in (1, size - 2, size - 1, size, size + 1, 3 * size):
        tiles = scan_windows(n, size, max(1, size // 2))
        pages, pairs = set(), set()
        assert sum(t[2] for t in tiles) == sum(t[3] for t in tiles) == 1
        for lo, hi, begin, end in tiles:
            pages.update(range(lo, hi))
            pairs.update(range(lo + 1, hi))
            assert (not begin or lo == 0) and (not end or hi == n)
            tokens = hi - lo + begin + end
            assert tokens <= size
            if tokens < size:
                assert begin and end
        assert pages == set(range(n))
        assert pairs == set(range(1, n))


def fixture(tmp_path):
    config = prepare_fixture(tmp_path, steps=8)
    config.update(scan_borders=True, scan_batches={'full_scan_multiplier': 4, 'stride': 6},
                  variable_lengths=True, auxiliary_edges=True)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = ScanBorderDataset(WindowDataset(index, features))
    backbone = BertBackbone.create(**{k: v for k, v in config['backbone'].items() if k != 'kind'})
    model = PageTagger(backbone, features.dims, auxiliary_edges=True, scan_borders=True)
    return config, index, dataset, model


def test_sampling_scan_contract_anchors_coverage_and_exact_resume(tmp_path):
    config, index, _, _ = fixture(tmp_path)
    sampler = ScanBatchSampler(index, 12, seed=17, variable_lengths=True,
                               scan_batches=config['scan_batches'])
    for stage in ('pretrain', 'finetune'):
        sampler.set_stage(stage)
        for _ in range(200):
            w = sampler.draw()
            pages, docs = index.pages_with_documents(w.rows)
            tokens = len(w.rows) + w.scan_start + w.scan_end
            assert tokens == 12 or (tokens < 12 and w.scan_start and w.scan_end)
            if w.scan_start:
                assert w.rows[0] == sampler.by_id[docs[0]].first
            if w.scan_end:
                d = sampler.by_id[docs[-1]]
                assert w.rows[-1] == d.first + d.length - 1
            if w.source == 'annotated':
                assert any(p.annotated_type for p in pages)
        clone = ScanBatchSampler(index, 12, variable_lengths=True, scan_batches=config['scan_batches'])
        clone.load_state_dict(sampler.state_dict())
        assert [sampler.draw() for _ in range(50)] == [clone.draw() for _ in range(50)]
    assert sampler.counters['scan_start_windows'] > 0
    assert sampler.counters['scan_end_windows'] > 0
    assert sampler.counters['internal_windows'] > 0
    coverage = ScanBatchSampler(index, 12)
    seen = set()
    while coverage.epoch == 0:
        seen.update(coverage._coverage().rows)
    assert seen == {r for d in coverage.docs for r in range(d.first, d.first + d.length)}
    changed = ScanBatchSampler(index, 12, variable_lengths=True, scan_batches={'stride': 3})
    with pytest.raises(ValueError, match='scan batch policy'):
        changed.load_state_dict(sampler.state_dict())


def test_marker_gradients_padding_invariance_dropout_and_checkpoint(tmp_path):
    _, index, dataset, model = fixture(tmp_path)
    d = index.documents('fit')[0]
    whole = dataset[Window(tuple(range(d.first, d.first + d.length)), scan_start=True, scan_end=True)]
    middle = dataset[Window((d.first + 3,), scan_start=False, scan_end=False)]
    batch = collate_windows([whole, middle])
    original = batch['features']['scan_borders'].clone()
    erased = scattered_dropout(batch['features'], modality_probability=1, page_probability=1)
    assert torch.equal(erased['scan_borders'], original)
    model.eval()
    output = model(batch['features'], batch['attention_mask'])
    assert output['type_logits'].shape == (2, 10, model.num_types)
    single = collate_windows([middle])
    separate = model(single['features'], single['attention_mask'])
    torch.testing.assert_close(output['type_logits'][1, :1], separate['type_logits'][0], atol=1e-6, rtol=1e-5)
    losses, _ = task_losses(output, batch)
    losses['total'].backward()
    assert (model.scan_border_embeddings.grad.abs().sum(1) > 0).all()
    restored = from_specification(model.specification())
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.scan_borders
    # Inference strips marker positions and retains every adjacent page pair.
    result = predict_stream(restored, dataset, tuple(range(d.first, d.first + d.length)),
                            InferenceConfig(window_size=12, stride=6))
    assert result['type_probs'].shape == (10, model.num_types)
    assert not result['pair_mask'][0] and result['pair_mask'][1:].all()


def test_real_runner_resumes_new_architecture_and_worker_dataset(tmp_path):
    config, _, _, _ = fixture(tmp_path)
    config.update(workers=2, worker_start_method='forkserver', batch_size=4,
                  final_scan_context_comparison=True,
                  context_evaluation={'enabled': True, 'sample_pages': 16, 'final_sample_pages': 16})
    train(config, stop_after=2)
    model, checkpoint = load_checkpoint(tmp_path / 'run/latest.pt')
    assert model.scan_borders and checkpoint['step'] == 2
    train(config, resume=tmp_path / 'run/latest.pt')
    records = [json.loads(s) for s in (tmp_path / 'run/validation.jsonl').read_text().splitlines()]
    assert records[-1]['step'] == 8
    assert records[-1]['dev']['curated_type']['n'] > 0
    comparison = list((tmp_path / 'run').glob('final-scan-context-step*/report.json'))
    assert len(comparison) == 1
    report = json.loads(comparison[0].read_text())
    assert report['modes']['per_document']['curated_type']['n'] == report['modes']['continuous']['curated_type']['n']
