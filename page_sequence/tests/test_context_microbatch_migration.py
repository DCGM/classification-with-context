import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def helper():
    spec = importlib.util.spec_from_file_location('change_microbatch',
        Path(__file__).resolve().parents[2] / 'tools/change_context_microbatch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_batch_migration_preserves_all_training_state_and_original(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    config = dict(window_size=1, context_microbatch_size=1024, sampling_window_size=512,
                  batch_size=64, run_dir=str(run))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'smoke-config.json').write_text(json.dumps(config))
    state = dict(config=config, step=3500, model={'w': torch.randn(4, 3)},
                 optimizer={'momentum': torch.randn(4)}, rng={'numpy': np.arange(5)},
                 sampler={'pending': [1, 3, 7]}, grad_scaler={'scale': 1024},
                 training={'stage': 'pretrain', 'bad_validations': 2})
    torch.save(state, run / 'latest.pt')
    original_bytes = (run / 'latest.pt').read_bytes()
    result = helper().change(tmp_path, 32768)
    changed = torch.load(run / 'latest.pt', weights_only=False)
    assert changed['config']['context_microbatch_size'] == 32768
    assert changed['config'] == json.loads((tmp_path / 'config.json').read_text())
    assert (Path(result['backup']) / 'latest.pt').read_bytes() == original_bytes
    for key in state:
        if key != 'config':
            helper().assert_same(state[key], changed[key])


def test_migration_refuses_unexpected_checkpoint_config(tmp_path):
    config = dict(window_size=1, context_microbatch_size=1024, sampling_window_size=512,
                  batch_size=64, run_dir=str(tmp_path / 'run'))
    (tmp_path / 'config.json').write_text(json.dumps(config))
    run = tmp_path / 'run'; run.mkdir()
    torch.save(dict(config=dict(config, batch_size=32)), run / 'latest.pt')
    before = (run / 'latest.pt').read_bytes()
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        helper().change(tmp_path, 32768)
    assert (run / 'latest.pt').read_bytes() == before
    assert json.loads((tmp_path / 'config.json').read_text()) == config
