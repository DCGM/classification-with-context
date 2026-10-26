import json
import sys
from types import SimpleNamespace

import pytest

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
