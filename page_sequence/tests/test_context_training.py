import copy
import json

import numpy as np
import pytest
import torch

from page_sequence.context_training import backward_chunks, split_context
from page_sequence.data import collate_windows
from page_sequence.inference import InferenceConfig, predict_stream
from page_sequence.losses import LossWeights, task_losses
from page_sequence.models import BertBackbone, PageTagger
from page_sequence.runner import make_dataset, train, load_checkpoint
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.schema import Window, IGNORE
from page_sequence.smoke import prepare_fixture


def setup_data(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    config['boundary_features'] = True
    index = PageIndex(config['index'])
    dataset = make_dataset(index, FeaturePipeline.from_config(config['features']), config)
    return config, index, dataset


def model_for(dataset):
    backbone = BertBackbone.create(pretrained=False, config={
        'vocab_size': 8, 'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 4,
        'intermediate_size': 32, 'max_position_embeddings': 128,
        'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.})
    return PageTagger(backbone, dataset.model_dims, auxiliary_edges=True)


@pytest.mark.parametrize('size', [1, 2, 4, 8])
def test_partition_preserves_pages_targets_weights_and_borders(tmp_path, size):
    _, index, ds = setup_data(tmp_path)
    d = index.documents('fit')[0]
    rows = tuple(range(d.first, d.first + 7))
    batch = collate_windows([ds[Window(rows, scan_start=True, scan_end=True)], ds[Window(rows[:3])]])
    chunks = split_context(batch, size)
    before = batch['attention_mask']
    after = chunks['attention_mask']
    assert after.any(1).all()
    for key in ('type', 'gold', 'placement', 'starts', 'page_start', 'page_end', 'weight'):
        torch.testing.assert_close(batch[key][before], chunks[key][after])
    for name in batch['features']:
        torch.testing.assert_close(batch['features'][name][before], chunks['features'][name][after])
    assert not chunks['pair_mask'][:, 0].any()
    assert not chunks['pair_weight'][:, 0].any()
    assert chunks['features']['boundaries'][..., 0].sum() == 1
    assert chunks['features']['boundaries'][..., 1].sum() == 1


@pytest.mark.parametrize('size', [1, 3])
def test_accumulation_matches_full_chunk_batch_and_disables_boundary_gradients(tmp_path, size):
    _, index, ds = setup_data(tmp_path)
    d = index.documents('fit')[0]
    batch = collate_windows([ds[Window(tuple(range(d.first, d.first + 8)))],
                             ds[Window(tuple(range(d.first, d.first + 3)))]])
    batch['gold'][0, :2] = True
    batch['type'][0, 3] = IGNORE
    model = model_for(ds).eval()
    reference = copy.deepcopy(model)
    weights = LossWeights(starts=0 if size == 1 else .5)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    values, denominators = backward_chunks(model, batch, batch['features'], size, 2, weights, scaler, 'cpu')
    chunks = split_context(batch, size)
    expected, expected_denoms = task_losses(reference(chunks['features'], chunks['attention_mask']), chunks, weights)
    expected['total'].backward()
    for key in values:
        torch.testing.assert_close(values[key], expected[key], atol=2e-6, rtol=2e-5)
    for key in expected_denoms:
        torch.testing.assert_close(denominators[key], expected_denoms[key])
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-4)
    if size == 1:
        assert all(p.grad is None for p in model.start_head.parameters())
        assert model.page_start_head.weight.grad.abs().sum() > 0


def test_singleton_inference_keeps_real_scan_flags_and_has_no_boundary_metrics(tmp_path):
    _, index, ds = setup_data(tmp_path)
    d = index.documents('dev')[0]
    rows = tuple(range(d.first, d.first + 7))
    model = model_for(ds).eval()
    results = [predict_stream(model, ds, rows, InferenceConfig(window_size=1, stride=1,
                           edge_ramp=1, batch_size=b), 'cpu') for b in [1, 4, 16]]
    for result in results:
        assert not result['pair_mask'].any()
        assert np.isnan(result['starts_document_probs']).all()
        assert result['boundary_enabled'] is False
        np.testing.assert_allclose(result['type_probs'], results[0]['type_probs'], atol=1e-6)
        assert result['processed_pages'] == len(rows)


def test_w1_train_phase_checkpoints_and_resume(tmp_path):
    config, _, _ = setup_data(tmp_path)
    config.update(window_size=1, sampling_window_size=12, context_microbatch_size=16,
                  auxiliary_edges=True, scan_batches={'stride': 6, 'border_tokens': False,
                  'metadata_probability': .5}, pretrain_steps=2, finetune_steps=2,
                  validate_every=2, document_metadata_evaluation=True)
    config['inference'] = dict(window_size=1, stride=1, edge_ramp=1, batch_size=16)
    config['loss_weights'] = dict(starts=0)
    train(config, stop_after=2)
    run = __import__('pathlib').Path(config['run_dir'])
    train(config, resume=run / 'latest.pt')
    for name in ['pretrain_final.pt', 'finetune_init.pt', 'finetune_final.pt']:
        assert (run / name).exists()
    model, state = load_checkpoint(run / 'latest.pt')
    assert state['sampler']['window_size'] == 12
    for line in (run / 'validation.jsonl').read_text().splitlines():
        report = json.loads(line)['dev']
        assert report['starts']['all']['n'] == 0
        assert report['starts']['all']['f1'] is None
        assert report['exact_stream_fraction'] is None
        assert report['curated_type']['n'] > 0


def test_failed_preflight_stops_full_training_and_unchanged_retry(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    import shutil
    source = Path(__file__).resolve().parents[2] / 'tools/context_window_job.py'
    shutil.copy2(source, tmp_path / 'launch.py')
    (tmp_path / 'code').mkdir()
    (tmp_path / 'config.json').write_text(json.dumps({'context_microbatch_size': 1024}))
    (tmp_path / 'smoke.py').write_text('raise RuntimeError("test failure")\n')
    for _ in range(2):
        result = subprocess.run([sys.executable, str(tmp_path / 'launch.py')], capture_output=True)
        assert result.returncode != 0
        assert b'preflight verification failed' in result.stderr
    assert json.loads((tmp_path / 'smoke-failed.json').read_text())['exit_code'] != 0
    assert not (tmp_path / 'smoke-passed.json').exists()


def test_preflight_does_not_reuse_a_smaller_physical_batch_check(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    import shutil
    shutil.copy2(Path(__file__).resolve().parents[2] / 'tools/context_window_job.py', tmp_path / 'launch.py')
    (tmp_path / 'config.json').write_text(json.dumps({'context_microbatch_size': 32768}))
    (tmp_path / 'smoke-passed.json').write_text(json.dumps({'context_microbatch_size': 1024}))
    result = subprocess.run([sys.executable, str(tmp_path / 'launch.py')], capture_output=True)
    assert result.returncode != 0
    assert b'physical batch changed' in result.stderr


def test_sampler_counters_are_logged_at_validations_only(tmp_path):
    config, _, _ = setup_data(tmp_path)
    config.update(pretrain_steps=3, finetune_steps=2, validate_every=2)
    train(config)
    run = __import__('pathlib').Path(config['run_dir'])
    steps = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert len(steps) == 5 and not any('sampler' in r for r in steps)
    validations = [json.loads(line) for line in (run / 'validation.jsonl').read_text().splitlines()]
    assert validations and all(v['sampler'] for v in validations)
    # A validation record is the update's metrics line plus the counters and the dev report.
    by_step = {r['step']: r for r in steps}
    for v in validations:
        assert {k: x for k, x in v.items() if k not in ('sampler', 'dev')} == by_step[v['step']]
    _, state = load_checkpoint(run / 'finetune_final.pt')
    assert validations[-1]['sampler'] == dict(state['sampler']['counters'])
