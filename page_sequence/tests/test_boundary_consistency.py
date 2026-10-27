import copy
import math
from pathlib import Path

import pytest
import torch

from page_sequence.losses import bernoulli_js, boundary_consistency
from page_sequence.runner import train, load_checkpoint
from page_sequence.schema import IGNORE
from page_sequence.smoke import prepare_fixture
from wait_training_slot import wait_for_slot


def example():
    output = {'starts_document_logits': torch.tensor([[0., 2., -2.]], requires_grad=True),
              'page_start_logits': torch.tensor([[-1., -2., 2.]], requires_grad=True),
              'page_end_logits': torch.tensor([[-2., 2., 1.]], requires_grad=True),
              'pair_mask': torch.tensor([[False, True, True]])}
    batch = {'attention_mask': torch.ones(1, 3, dtype=torch.bool),
             'pair_mask': output['pair_mask'].clone(), 'pair_weight': torch.ones(1, 3),
             'starts': torch.tensor([[0., 1., 0.]]),
             'page_start': torch.tensor([[1., 1., 0.]]),
             'page_end': torch.tensor([[1., 0., 1.]])}
    return output, batch


def test_js_equal_symmetric_bounded_and_fp16_extremes():
    a = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float16, requires_grad=True)
    b = -a
    torch.testing.assert_close(bernoulli_js(a, a), torch.zeros(5), atol=1e-7, rtol=0)
    torch.testing.assert_close(bernoulli_js(a, b), bernoulli_js(b, a))
    js = bernoulli_js(a, b)
    assert torch.isfinite(js).all() and (js >= 0).all() and (js <= math.log(2) + 1e-6).all()
    js.sum().backward()
    assert torch.isfinite(a.grad).all()


def test_alignment_gradients_and_absent_outer_neighbours():
    output, batch = example()
    loss, denom, conflicts = boundary_consistency(output, batch)
    assert loss > 0 and denom == 2 and conflicts == 0
    loss.backward()
    assert output['starts_document_logits'].grad[0, 1:].abs().sum() > 0
    assert output['page_start_logits'].grad[0, 1:].abs().sum() > 0
    assert output['page_end_logits'].grad[0, :-1].abs().sum() > 0
    assert output['page_start_logits'].grad[0, 0] == 0
    assert output['page_end_logits'].grad[0, -1] == 0
    del output['page_end_logits']
    with pytest.raises(ValueError, match='auxiliary'):
        boundary_consistency(output, batch)


def test_unknown_conflicting_padding_and_singleton_masks():
    output, batch = example()
    batch['page_end'][0, 0] = 0  # conflicts with the boundary/start labels on page 1
    loss, denom, conflicts = boundary_consistency(output, batch)
    assert denom == 1 and conflicts == 1
    batch['page_start'][0, 2] = IGNORE
    loss, denom, conflicts = boundary_consistency(output, batch)
    assert denom == 0 and conflicts == 1 and loss == 0
    output, batch = example()
    batch['attention_mask'][0, 2] = False
    loss, denom, conflicts = boundary_consistency(output, batch)
    assert denom == 1
    for name in output:
        output[name] = output[name][:, :1]
    for name in batch:
        batch[name] = batch[name][:, :1]
    loss, denom, conflicts = boundary_consistency(output, batch)
    assert loss == 0 and denom == 0 and conflicts == 0


def test_consistency_training_resume(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=8)
    config.update(auxiliary_edges=True, loss_weights={'boundary_consistency': .1},
                  edge_crops={'start_probability': .1, 'end_probability': .1})
    train(config)
    _, original = load_checkpoint(Path(config['run_dir']) / 'latest.pt')
    resumed = copy.deepcopy(config)
    resumed['run_dir'] = str(tmp_path / 'resumed')
    train(resumed, stop_after=4)
    path = Path(resumed['run_dir']) / 'latest.pt'
    train(resumed, resume=path)
    _, restored = load_checkpoint(path)
    for key, value in original['model'].items():
        torch.testing.assert_close(value, restored['model'][key], rtol=0, atol=0)


def test_slot_gate_fails_closed_and_releases_successful_run(tmp_path):
    import json
    status = tmp_path / 'blocking-status.json'
    status.write_text(json.dumps({'jobs': {}}))
    with pytest.raises(ValueError, match='no jobs'):
        wait_for_slot(tmp_path, status, poll_seconds=0)
    status.write_text(json.dumps({'jobs': {'ssd': {'state': 'failed', 'exit_code': 1}}}))
    with pytest.raises(RuntimeError, match='failed'):
        wait_for_slot(tmp_path, status, poll_seconds=0)
    status.write_text(json.dumps({'jobs': {'ssd': {'state': 'finished', 'exit_code': 0}}}))
    wait_for_slot(tmp_path, status, poll_seconds=0)
