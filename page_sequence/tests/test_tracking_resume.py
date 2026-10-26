import json
import sys
import random
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from page_sequence.tracking import Tracking


class FakeTask:
    calls=[]
    id='saved-task-id'

    @classmethod
    def init(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls()

    def connect_configuration(self, *args, **kwargs):
        pass

    def get_logger(self):
        return SimpleNamespace()

    def get_output_log_web_page(self):
        return 'http://clearml.example/task'


@pytest.mark.parametrize('resume',[False,True])
def test_resume_preserves_task_without_adding_iteration_offset(tmp_path,monkeypatch,resume):
    FakeTask.calls=[]
    monkeypatch.setitem(sys.modules,'clearml',SimpleNamespace(Task=FakeTask))
    if resume:
        (tmp_path/'clearml.json').write_text(json.dumps({'id':'saved-task-id'}))
    Tracking({'run_dir':str(tmp_path),'clearml':{'enabled':True}}, {}, resume=resume)
    args=FakeTask.calls[-1]
    if resume:
        assert args['reuse_last_task_id']=='saved-task-id'
        assert type(args['continue_last_task']) is int
        assert args['continue_last_task']==0
    else:
        assert args['reuse_last_task_id'] is False
        assert args['continue_last_task'] is False
    assert json.loads((tmp_path/'clearml.json').read_text())['id']=='saved-task-id'


def test_resume_missing_task_identity_fails_without_creating_new_task(tmp_path,monkeypatch):
    FakeTask.calls=[]
    monkeypatch.setitem(sys.modules,'clearml',SimpleNamespace(Task=FakeTask))
    with pytest.raises(FileNotFoundError,match='ClearML history'):
        Tracking({'run_dir':str(tmp_path),'clearml':{'enabled':True}}, {}, resume=True)
    assert not FakeTask.calls


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_sdk_reseeding_does_not_change_resumed_training(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from page_sequence import runner
    from page_sequence.smoke import prepare_fixture

    class ReseedingTracking:
        def __init__(self, *args):
            random.seed(1337)
            np.random.seed(1337)
            torch.manual_seed(1337)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1337)
        def report(self, *args, **kwargs):
            pass
        def close(self, *args, **kwargs):
            pass

    monkeypatch.setattr(runner, 'Tracking', ReseedingTracking)
    config = prepare_fixture(tmp_path / 'fixture', steps=8)
    config.update(device=device, scattered_dropout={'modality_probability': .1, 'page_probability': .05},
                  precision='float16' if device == 'cuda' else 'float32')
    runner.train(config)
    _, original = runner.load_checkpoint(Path(config['run_dir']) / 'latest.pt')
    resumed_config = copy.deepcopy(config)
    resumed_config['run_dir'] = str(tmp_path / 'resumed')
    runner.train(resumed_config, stop_after=4)  # interrupt at the existing validation boundary
    path = Path(resumed_config['run_dir']) / 'latest.pt'
    runner.train(resumed_config, resume=path)
    _, resumed = runner.load_checkpoint(path)
    for key, value in original['model'].items():
        torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
