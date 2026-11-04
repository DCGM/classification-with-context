import importlib.util
import json
from pathlib import Path


def module(name):
    path = Path(__file__).resolve().parents[2] / 'tools' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_failure_policy_and_bounded_backoff():
    monitor = module('training_monitor')
    assert monitor.failure_policy('CUDA out of memory') == 'retry'
    assert monitor.failure_policy('DataLoader worker exited unexpectedly') == 'retry'
    assert monitor.failure_policy('resume config/data/code/environment/feature schema differs') == 'needs_diagnosis'
    assert monitor.failure_policy('OSError: No space left on device') == 'needs_diagnosis'
    assert monitor.retry_delay(0) == 60
    assert monitor.retry_delay(100) == 1800


def test_progress_ignores_incomplete_last_write(tmp_path):
    monitor = module('training_monitor')
    (tmp_path / 'metrics.jsonl').write_text('{"step": 17}\n{"step":')
    assert monitor.progress(tmp_path) == 17


def test_retry_blocks_repeated_failures_without_progress(tmp_path):
    monitor = module('training_monitor')
    root = tmp_path / 'runtime'; root.mkdir()
    run = tmp_path / 'run'; run.mkdir()
    (root/'config.json').write_text(json.dumps({'run_dir':str(run)}))
    (root/'training.log').write_text('worker exited unexpectedly')
    (run/'metrics.jsonl').write_text('{"step": 100}\n')
    job = {'id':'a', 'root':str(root)}
    m = object.__new__(monitor.Monitor)
    m.root = tmp_path
    m.state = {'jobs':{'a':{'attempts':0}},'events':[]}
    m.failed(job); assert m.state['jobs']['a']['phase'] == 'retry_wait'
    m.failed(job); assert m.state['jobs']['a']['phase'] == 'retry_wait'
    m.failed(job); assert m.state['jobs']['a']['phase'] == 'needs_diagnosis'
    (run/'metrics.jsonl').write_text('{"step": 200}\n')
    m.failed(job); assert m.state['jobs']['a']['same_failure_count'] == 1


def test_attempt_records_exit_and_passes_resume(tmp_path):
    import subprocess,sys
    (tmp_path/'launch.py').write_text('import sys\nassert sys.argv[1:] == ["--resume"]\n')
    script=Path(__file__).resolve().parents[2]/'tools/training_attempt.py'
    subprocess.run([sys.executable,str(script),'--root',str(tmp_path),'--resume'],check=True)
    assert json.loads((tmp_path/'managed-exit.json').read_text())['exit_code']==0


def test_three_job_limit_includes_external_training(tmp_path, monkeypatch):
    monitor = module('training_monitor')
    runtime=tmp_path/'tail'; runtime.mkdir()
    run=tmp_path/'run'; run.mkdir()
    (run/'fork.json').write_text('{}')
    (runtime/'config.json').write_text(json.dumps({'run_dir':str(run)}))
    job={'id':'tail','kind':'tail','root':str(runtime),'unit':'tail.service'}
    (tmp_path/'monitor-config.json').write_text(json.dumps({'max_trainings':3,'jobs':[job]}))
    m=monitor.Monitor(tmp_path); m.state['jobs']['tail']['phase']='ready'
    monkeypatch.setattr(m,'active',lambda job:False)
    monkeypatch.setattr(monitor,'available_gib',lambda:10)
    monkeypatch.setattr(monitor,'training_processes',lambda:{1:'/a',2:'/b',3:'/c'})
    launches=[]; monkeypatch.setattr(m,'launch',lambda job:launches.append(job))
    m.tick()
    assert not launches and m.state['active_trainings']==3
    monkeypatch.setattr(monitor,'training_processes',lambda:{1:'/a',2:'/b'})
    m.tick()
    assert len(launches)==1


def test_completed_parent_reconstruction_preserves_best_pretrained_weights(tmp_path):
    import torch
    helper=module('training_tail_helper')
    parent=tmp_path/'parent'; parent.mkdir()
    run=tmp_path/'run'; run.mkdir()
    target=tmp_path/'tail'; target.mkdir()
    (parent/'config.json').write_text(json.dumps({'run_dir':str(run)}))
    torch.save({'step':140,'training':{'finished':True,'stage_step':40}},run/'latest.pt')
    source={'step':90,'training':{'stage':'pretrain'},'sampler':{'stage':'pretrain','pending_sources':['coverage'],'rng':{'state':123}},'model':{'weight':torch.tensor([42.])},'optimizer':{'momentum':torch.tensor([5.])},'rng':{'state':999}}
    torch.save(source,run/'pretrain_best.pt')
    result=torch.load(helper.reconstruct(parent,target),weights_only=False)
    assert result['step']==100 and result['training']['stage_step']==0
    assert result['sampler']['stage']=='finetune' and not result['sampler']['pending_sources']
    assert result['sampler']['rng']==source['sampler']['rng'] and result['rng']==source['rng']
    assert torch.equal(result['model']['weight'],source['model']['weight'])
    assert torch.equal(result['optimizer']['momentum'],source['optimizer']['momentum'])
    assert not json.loads((target/'transition-reconstruction.json').read_text())['exact_parent_transition']


def test_launch_waits_for_configured_memory_headroom(tmp_path, monkeypatch):
    monitor = module('training_monitor')
    root = tmp_path / 'runtime'; root.mkdir()
    (root / 'config.json').write_text(json.dumps({'run_dir': str(tmp_path / 'run')}))
    job = {'id': 'a', 'kind': 'parent', 'root': str(root), 'unit': 'a.service'}
    (tmp_path / 'monitor-config.json').write_text(json.dumps({
        'max_trainings': 2, 'min_launch_available_gib': 9, 'jobs': [job]}))
    m = monitor.Monitor(tmp_path)
    monkeypatch.setattr(m, 'active', lambda job: False)
    monkeypatch.setattr(monitor, 'training_processes', lambda: {})
    monkeypatch.setattr(monitor, 'available_gib', lambda: 8)
    launches = []
    monkeypatch.setattr(m, 'launch', lambda job: launches.append(job))
    m.tick()
    assert not launches
    monkeypatch.setattr(monitor, 'available_gib', lambda: 10)
    m.tick()
    assert len(launches) == 1


def test_queue_reserves_one_job_per_storage_slot(tmp_path, monkeypatch):
    monitor = module('training_monitor')
    jobs = []
    for name, slot in [('S', 'nvme'), ('I', 'nvme'), ('T', 'ssd')]:
        root = tmp_path / name; root.mkdir()
        (root / 'config.json').write_text(json.dumps({'run_dir': str(root / 'run')}))
        jobs.append(dict(id=name, kind='parent', root=str(root), unit=name + '.service', slot=slot))
    (tmp_path / 'monitor-config.json').write_text(json.dumps({'max_trainings': 2, 'jobs': jobs}))
    m = monitor.Monitor(tmp_path)
    monkeypatch.setattr(m, 'active', lambda job: False)
    monkeypatch.setattr(monitor, 'available_gib', lambda: 20)
    monkeypatch.setattr(monitor, 'training_processes', lambda: {})
    launched = []
    def launch(job):
        launched.append(job['id'])
        m.state['jobs'][job['id']].update(phase='running', launched=monitor.time.time())
    monkeypatch.setattr(m, 'launch', launch)
    m.tick()
    assert launched == ['S', 'T']


def test_queue_waits_for_completed_dependency(tmp_path, monkeypatch):
    monitor = module('training_monitor')
    jobs = []
    for name in ['parent', 'next']:
        root = tmp_path / name; root.mkdir()
        (root / 'config.json').write_text(json.dumps({'run_dir': str(root / 'run')}))
        jobs.append(dict(id=name, kind='parent', root=str(root), unit=name + '.service'))
    jobs[1]['depends_on'] = ['parent']
    (tmp_path / 'monitor-config.json').write_text(json.dumps({'max_trainings': 2, 'jobs': jobs}))
    m = monitor.Monitor(tmp_path)
    m.state['jobs']['parent']['phase'] = 'retry_wait'
    m.state['jobs']['parent']['next_attempt'] = monitor.time.time() + 10000
    monkeypatch.setattr(m, 'active', lambda job: False)
    monkeypatch.setattr(monitor, 'available_gib', lambda: 20)
    monkeypatch.setattr(monitor, 'training_processes', lambda: {})
    launches = []
    monkeypatch.setattr(m, 'launch', lambda job: launches.append(job['id']))
    m.tick()
    assert not launches
    m.state['jobs']['parent']['phase'] = 'finished'
    m.tick()
    assert launches == ['next']
