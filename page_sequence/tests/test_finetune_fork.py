import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from fork_finetune import create_fork, check_config, check_provenance, require_transition
from watch_finetune_fork import capture_transition as watch_transition
from page_sequence import runner
from page_sequence.checkpoint_migration import protected_digest
from page_sequence.smoke import prepare_fixture


def capture_transition(tmp_path, monkeypatch):
    config = prepare_fixture(tmp_path / 'fixture', steps=8)
    config.update(auxiliary_edges=True, edge_crops={'start_probability': .1, 'end_probability': .1})
    original = runner.save_checkpoint
    snapshot = tmp_path / 'transition.pt'
    def save(path, model, optimizer, sampler, step, best, settings, prov, features, training_state=None):
        original(path, model, optimizer, sampler, step, best, settings, prov, features, training_state)
        if (Path(path).name == 'latest.pt' and training_state['stage'] == 'finetune'
                and training_state['stage_step'] == 0):
            shutil.copyfile(path, snapshot)
    monkeypatch.setattr(runner, 'save_checkpoint', save)
    runner.train(config)
    return config, snapshot


def test_fork_preserves_actual_transition_and_only_changes_fine_loss(tmp_path, monkeypatch):
    parent, snapshot = capture_transition(tmp_path, monkeypatch)
    target = copy.deepcopy(parent)
    target.update(run_dir=str(tmp_path / 'fork'), stage_loss_weights={'finetune': {'weak_type': .05}})
    checkpoint = create_fork(snapshot, target)
    before = torch.load(snapshot, weights_only=False)
    after = torch.load(checkpoint, weights_only=False)
    assert after['training']['stage'] == 'finetune' and after['training']['stage_step'] == 0
    assert after['step'] == before['step'] == parent['pretrain_steps']
    assert after['sampler'] == before['sampler']
    normalized = dict(after, config=before['config'], provenance=before['provenance'])
    assert protected_digest(before) == protected_digest(normalized)
    runner.train(target, resume=checkpoint)
    _, result = runner.load_checkpoint(Path(target['run_dir']) / 'latest.pt')
    assert result['training']['finished']
    def metrics(config):
        return [json.loads(line) for line in (Path(config['run_dir']) / 'metrics.jsonl').read_text().splitlines()]
    a = next(row for row in metrics(parent) if row['stage'] == 'finetune')
    child = metrics(target)
    assert all(row['stage'] == 'finetune' for row in child)
    assert child[0]['loss_weights']['weak_type'] == .05
    for task in a['loss']:
        if task != 'total':
            assert a['loss'][task] == child[0]['loss'][task]
    assert (Path(target['run_dir']) / 'fork.json').exists()
    with pytest.raises(FileExistsError):
        create_fork(snapshot, target)


def test_fork_rejects_unrelated_config_or_provenance_changes(tmp_path):
    parent = {'run_dir': str(tmp_path / 'parent'), 'seed': 17}
    target = dict(parent, run_dir=str(tmp_path / 'child'), stage_loss_weights={'finetune': {'weak_type': .05}})
    check_config(parent, target)
    with pytest.raises(ValueError, match='seed'):
        check_config(parent, dict(target, seed=18))
    with pytest.raises(ValueError, match='pretraining'):
        check_config(parent, dict(target, stage_loss_weights={'pretrain': {'weak_type': .05},
                                                             'finetune': {'weak_type': .05}}))
    prov = {'code_sha256': 'same', 'training_store': {'path': '/nvme', 'build_id': 'same'}}
    check_provenance(prov, {'code_sha256': 'same', 'training_store': {'path': '/ssd', 'build_id': 'same'}})
    with pytest.raises(ValueError, match='provenance'):
        check_provenance(prov, dict(prov, code_sha256='changed'))
    with pytest.raises(ValueError, match='provenance'):
        check_provenance(prov, {'code_sha256': 'same', 'training_store': {'path': '/ssd', 'build_id': 'different'}})


def test_fork_refuses_pretrain_best_or_already_started_finetune(tmp_path, monkeypatch):
    config, snapshot = capture_transition(tmp_path, monkeypatch)
    transition = torch.load(snapshot, weights_only=False)
    require_transition(transition)
    already_started = copy.deepcopy(transition)
    already_started['training']['stage_step'] = 1
    with pytest.raises(ValueError, match='step-zero'):
        require_transition(already_started)
    pre = torch.load(Path(config['run_dir']) / 'pretrain_best.pt', weights_only=False)
    with pytest.raises(ValueError, match='step-zero'):
        require_transition(pre)


def test_watcher_captures_exact_atomic_transition(tmp_path, monkeypatch):
    config, source = capture_transition(tmp_path, monkeypatch)
    parent, target = tmp_path / 'parent-root', tmp_path / 'target-root'
    parent.mkdir()
    target.mkdir()
    (parent / 'config.json').write_text(json.dumps(config))
    (parent / 'status.json').write_text(json.dumps({'jobs': {'parent': {'state': 'running'}}}))
    shutil.copyfile(source, Path(config['run_dir']) / 'latest.pt')
    snapshot = watch_transition(parent, target, poll_seconds=0)
    assert snapshot.read_bytes() == source.read_bytes()
    assert json.loads((target / 'pipeline-status.json').read_text())['phase'] == 'captured_transition'
