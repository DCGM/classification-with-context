import pytest
import torch

from page_sequence.runner import load_checkpoint, train
from page_sequence.smoke import prepare_fixture


def test_missing_task_zero_loss_cannot_overflow_fp16():
    from page_sequence.losses import task_losses
    logits = torch.full((1, 128, 39), 100., dtype=torch.float16, requires_grad=True)
    output = {'type_logits': logits, 'placement_logits': logits[..., :3],
              'starts_document_logits': logits[..., 0],
              'pair_mask': torch.zeros(1, 128, dtype=torch.bool)}
    batch = {'attention_mask': torch.ones(1, 128, dtype=torch.bool),
             'type': torch.full((1, 128), -100), 'gold': torch.zeros(1, 128, dtype=torch.bool),
             'placement': torch.full((1, 128), -100), 'starts': torch.zeros(1, 128),
             'pair_mask': output['pair_mask'], 'weight': torch.ones(1, 128),
             'pair_weight': torch.ones(1, 128)}
    losses, _ = task_losses(output, batch)
    assert losses['total'].item() == 0
    losses['total'].backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FP16 requires CUDA')
def test_fp16_training_evaluation_and_resume(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    config.update(device='cuda', precision='float16', deterministic=False)
    train(config, stop_after=1)
    model, first = load_checkpoint(tmp_path / 'run/latest.pt', 'cuda')
    assert first['step'] == 1
    assert first['grad_scaler']['scale'] > 0
    assert model.compute_precision == 'float16'
    assert all(p.dtype == torch.float32 for p in model.parameters())
    del model
    train(config, resume=tmp_path / 'run/latest.pt')
    model, last = load_checkpoint(tmp_path / 'run/latest.pt')
    assert model.compute_precision == 'float32'
    assert last['step'] == 4
    assert last['training']['finished']
    assert all(torch.isfinite(v).all() for v in last['model'].values())
    assert any(not torch.equal(first['model'][k], v) for k, v in last['model'].items())
    from page_sequence.data import WindowDataset
    from page_sequence.features import FeaturePipeline
    from page_sequence.index import PageIndex
    from page_sequence.inference import InferenceConfig, predict_stream
    dataset = WindowDataset(PageIndex(config['index']), FeaturePipeline.from_config(config['features']))
    result = predict_stream(model, dataset, tuple(range(1, 21)),
                            InferenceConfig(window_size=12, stride=6))
    import numpy as np
    assert np.isfinite(result['type_probs']).all()
    assert np.isfinite(result['starts_document_probs'][1:]).all()
