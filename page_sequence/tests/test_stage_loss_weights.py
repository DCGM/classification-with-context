import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from page_sequence.losses import stage_loss_weights
from page_sequence.runner import load_checkpoint, train
from page_sequence.smoke import prepare_fixture


def test_stage_overrides_change_only_requested_task():
    config = {'loss_weights': {'weak_type': .2, 'placement': .7},
              'stage_loss_weights': {'finetune': {'weak_type': .05}}}
    pre = asdict(stage_loss_weights(config, 'pretrain'))
    fine = asdict(stage_loss_weights(config, 'finetune'))
    assert pre['weak_type'] == .2 and fine['weak_type'] == .05
    assert {k: v for k, v in pre.items() if k != 'weak_type'} == {
        k: v for k, v in fine.items() if k != 'weak_type'}
    assert fine['placement'] == .7 and fine['gold_type'] == 1


@pytest.mark.parametrize('config', [
    {'stage_loss_weights': {'unknown': {}}},
    {'stage_loss_weights': {'finetune': {'weak_type': -1}}},
    {'stage_loss_weights': {'finetune': {'weak_type': float('nan')}}},
    {'stage_loss_weights': {'finetune': {'unknown': 1}}},
])
def test_invalid_stage_weights_rejected(config):
    with pytest.raises(ValueError):
        stage_loss_weights(config, 'finetune')


def test_pretraining_unchanged_finetune_weights_and_resume(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=8)
    config.update(auxiliary_edges=True, edge_crops={'start_probability': .1, 'end_probability': .1})
    train(config)
    modified = copy.deepcopy(config)
    modified.update(run_dir=str(tmp_path / 'weak005'), stage_loss_weights={'finetune': {'weak_type': .05}})
    train(modified)
    for name in ('pretrain_best.pt',):
        _, a = load_checkpoint(Path(config['run_dir']) / name)
        _, b = load_checkpoint(Path(modified['run_dir']) / name)
        for key in a['model']:
            torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    metrics = [[json.loads(line) for line in (Path(c['run_dir']) / 'metrics.jsonl').read_text().splitlines()]
               for c in (config, modified)]
    a, b = [next(r for r in records if r['stage'] == 'finetune') for records in metrics]
    assert a['loss_weights']['weak_type'] == .2 and b['loss_weights']['weak_type'] == .05
    for task in a['loss']:
        if task != 'total':
            assert a['loss'][task] == b['loss'][task]
    assert a['loss']['total'] - b['loss']['total'] == pytest.approx(.15 * b['loss']['weak_type'], abs=1e-6)
    assert all(r['loss_weights']['weak_type'] == .2 for r in metrics[1] if r['stage'] == 'pretrain')
    resumed = copy.deepcopy(modified)
    resumed['run_dir'] = str(tmp_path / 'resumed')
    train(resumed, stop_after=4)
    path = Path(resumed['run_dir']) / 'latest.pt'
    train(resumed, resume=path)
    _, original = load_checkpoint(Path(modified['run_dir']) / 'latest.pt')
    _, restored = load_checkpoint(path)
    for key in original['model']:
        torch.testing.assert_close(original['model'][key], restored['model'][key], rtol=0, atol=0)
