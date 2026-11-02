"""Persistent, bounded-concurrency queue for local training and fine-tuning tails.

The daemon is stdlib-only. Checkpoint work runs in one short-lived helper at a
 time, keeping Torch out of the monitor and preserving frozen trainer identities.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def load_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    raise RuntimeError('cannot read available memory')


def training_processes():
    result = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            words = [x.decode(errors='replace') for x in args]
            if 'page_sequence.runner' in words and 'train' in words and '--config' in words:
                result[int(p.name)] = str(Path(words[words.index('--config') + 1]).resolve())
        except (OSError, ValueError, IndexError):
            pass
    return result


def tail(path, limit=65536):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def progress(run):
    for line in reversed(tail(run / 'metrics.jsonl').splitlines()):
        try:
            record = json.loads(line)
            return record['step']
        except (ValueError, KeyError):
            continue
    return 0


def failure_policy(text):
    # Resume cannot repair these conditions. Preserve evidence and request diagnosis.
    deterministic = ('resume config/data/code/environment/feature schema differs',
                     'checkpoint/index mismatch', 'checkpoint/training-store identity mismatch',
                     'pagination sidecar changed', 'No space left on device',
                     'ModuleNotFoundError', 'SyntaxError', 'PermissionError',
                     'nonfinite training loss', 'repeated FP16 gradient overflow',
                     'invalid device ordinal')
    if any(token in text for token in deterministic):
        return 'needs_diagnosis'
    return 'retry'


def retry_delay(attempts):
    return min(1800, 60 * 2 ** min(attempts, 5))


class Monitor:
    def __init__(self, root):
        self.root = root
        self.config = load_json(root / 'monitor-config.json')
        self.state = load_json(root / 'state.json', {'jobs': {}, 'events': []})
        self.helper = None
        self.helper_log = None
        self.helper_started = None
        self.low_since = None
        self.stopping = {}
        interrupted = self.state.get('helper')
        if interrupted and interrupted.get('mode') == 'inspect':
            self.state['jobs'].get(interrupted['job'], {}).pop('inspected_stamp', None)
        for job in self.config['jobs']:
            self.state['jobs'].setdefault(job['id'], {'attempts': 0, 'phase': 'queued'})

    def event(self, job, message, **fields):
        entry = dict(time=datetime.datetime.now().astimezone().isoformat(), job=job, message=message, **fields)
        with (self.root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        self.state['events'] = (self.state['events'] + [entry])[-50:]
        print(json.dumps(entry), flush=True)

    def active(self, job):
        units = [job['unit'], job.get('original_unit')]
        return any(subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit],
                                 check=False).returncode == 0 for unit in units if unit)

    def start_helper(self, job, mode):
        if self.helper is not None or available_gib() < 2.5:
            return False
        root = Path(job['root'])
        output = root / (mode + '-result.json')
        output.unlink(missing_ok=True)
        env = os.environ | {'PYTHONPATH': str(root / 'code') + ':' + str(self.root),
                            'HF_HOME': '/home/ikohut/tmp/huggingface',
                            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        args = [self.config['python'], str(self.root / 'training_tail_helper.py'), mode,
                '--parent', job['parent_root'], '--target', str(root), '--output', str(output)]
        if job.get('allow_reconstruction'):
            args += ['--allow-reconstruction']
        self.helper_log = (root / (mode + '.log')).open('ab', buffering=0)
        proc = subprocess.Popen(args, cwd=root / 'code', env=env,
                                stdout=self.helper_log, stderr=subprocess.STDOUT)
        self.helper = (proc, job, mode, output)
        self.helper_started = time.time()
        return True

    def collect_helper(self):
        if self.helper is None:
            return
        proc, job, mode, output = self.helper
        if proc.poll() is None:
            if time.time() - self.helper_started > 1800:
                proc.kill()
                self.event(job['id'], 'checkpoint helper exceeded 30 minutes')
            return
        self.helper_log.close()
        self.helper_log = None
        state = self.state['jobs'][job['id']]
        if proc.returncode:
            state['phase'] = 'needs_diagnosis'
            state['helper_error'] = tail(Path(job['root']) / (mode + '.log'), 4096)
            self.event(job['id'], 'checkpoint helper failed; preserved for diagnosis', mode=mode)
        else:
            result = load_json(output)
            state[mode] = result
            if mode == 'prepare':
                state['phase'] = 'ready'
                self.event(job['id'], 'fine-tuning fork prepared', exact_transition=result.get('exact_transition'))
            elif result.get('snapshot'):
                state['phase'] = 'captured'
                self.event(job['id'], 'captured exact fine-tuning initialization')
        self.helper = None

    def launch(self, job):
        root = Path(job['root'])
        run = Path(load_json(root / 'config.json')['run_dir'])
        latest = run / 'latest.pt'
        if job['kind'] == 'parent' and (run / 'metrics.jsonl').exists() and not latest.exists():
            self.state['jobs'][job['id']]['phase'] = 'needs_diagnosis'
            self.event(job['id'], 'training history exists but no resume checkpoint')
            return
        log = root / ('managed-training.log')
        (root / 'managed-exit.json').unlink(missing_ok=True)
        command = [self.config['python'], '-u', str(self.root / 'training_attempt.py'), '--root', str(root)]
        if latest.exists() or job['kind'] == 'tail':
            command.append('--resume')
        args = ['systemd-run', '--user', '--quiet', '--collect', '--unit=' + job['unit'],
                '--property=WorkingDirectory=' + str(root), '--property=KillMode=control-group',
                '--property=TimeoutStopSec=45', '--property=StandardOutput=append:' + str(log),
                '--property=StandardError=append:' + str(log)] + command
        result = subprocess.run(args, capture_output=True, text=True)
        state = self.state['jobs'][job['id']]
        if result.returncode:
            state['next_attempt'] = time.time() + 60
            self.event(job['id'], 'systemd launch failed', error=result.stderr.strip())
            return
        state.update(phase='running', managed=True, launched=time.time(), last_step=progress(run))
        self.event(job['id'], 'started training', resume=latest.exists() or job['kind'] == 'tail')

    def completion(self, job):
        root = Path(job['root'])
        if self.state['jobs'][job['id']].get('managed'):
            # Collected units disappear, so retain completion via launch wrapper markers.
            marker = load_json(root / 'managed-exit.json')
            return marker is not None and marker.get('exit_code') == 0
        status = load_json(root / 'status.json', {})
        return bool(status.get('jobs')) and all(j.get('exit_code') == 0
                                               for j in status['jobs'].values())

    def failed(self, job):
        state = self.state['jobs'][job['id']]
        root = Path(job['root'])
        cfg = load_json(root / 'config.json')
        step = progress(Path(cfg['run_dir']))
        text = tail(root / ('managed-training.log' if state.get('managed') else 'training.log'), 12000)
        traceback = text[text.rfind('Traceback'):] if 'Traceback' in text else text[-2000:]
        fingerprint = hashlib.sha256(traceback.encode()).hexdigest()
        repeated = state.get('same_failure_count', 0) + 1 if step <= state.get('failure_step', -1) else 1
        state.update(failure_fingerprint=fingerprint, same_failure_count=repeated,
                     failure_step=step, attempts=state.get('attempts', 0)+1,
                     next_attempt=time.time()+retry_delay(state.get('attempts', 0)))
        state['phase'] = 'needs_diagnosis' if failure_policy(text) == 'needs_diagnosis' or repeated >= 3 else 'retry_wait'
        atomic_json(root / 'last-failure.json', dict(step=step, log_tail=traceback, **state))
        self.event(job['id'], 'training stopped', phase=state['phase'], step=step, repeated=repeated)

    def tick(self):
        self.collect_helper()
        processes = training_processes()
        active_jobs = []
        for job in self.config['jobs']:
            state = self.state['jobs'][job['id']]
            if self.active(job) or str(Path(job['root']) / 'config.json') in processes.values():
                active_jobs.append(job)
                if state['phase'] != 'running':
                    state['phase'] = 'running'
                    state.setdefault('launched', time.time())
                run = Path(load_json(Path(job['root']) / 'config.json')['run_dir'])
                metrics = run / 'metrics.jsonl'
                changed = max(state.get('launched', time.time()), metrics.stat().st_mtime if metrics.exists() else 0)
                if time.time() - changed > 1800 and job['id'] not in self.stopping:
                    unit = job['unit'] if state.get('managed') else job.get('original_unit', job['unit'])
                    subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=main', '--signal=SIGINT', unit], check=False)
                    # Original main process is a supervisor; interrupt its trainer directly too.
                    for pid, config_path in processes.items():
                        if config_path == str(Path(job['root']) / 'config.json'):
                            try: os.kill(pid, 2)
                            except ProcessLookupError: pass
                    self.stopping[job['id']] = time.time()
                    self.event(job['id'], 'no training progress for 30 minutes; interrupting for resume')
                continue
            if state['phase'] == 'running':
                if self.completion(job):
                    state['phase'] = 'finished'
                    self.event(job['id'], 'finished successfully')
                else:
                    self.failed(job)
            elif job['kind'] == 'parent' and state['phase'] == 'queued' and self.completion(job):
                state['phase'] = 'finished'
        # Memory pressure: stop only the newest managed attempt before old guards fire.
        memory = available_gib()
        self.low_since = (self.low_since or time.time()) if memory < 3 else None
        if self.low_since and time.time() - self.low_since >= 10:
            candidates = [j for j in active_jobs if self.state['jobs'][j['id']].get('managed')]
            if candidates:
                job = max(candidates, key=lambda j:self.state['jobs'][j['id']].get('launched', 0))
                if job['id'] not in self.stopping:
                    subprocess.run(['systemctl','--user','kill','--kill-whom=main','--signal=SIGINT',job['unit']],check=False)
                    self.stopping[job['id']] = time.time()
                    self.event(job['id'], 'memory guard interrupted newest managed training')
        for jid, when in list(self.stopping.items()):
            job = next(j for j in self.config['jobs'] if j['id']==jid)
            if not self.active(job):
                del self.stopping[jid]
                self.state['jobs'][jid]['next_attempt'] = time.time()+300
            elif time.time()-when > 45:
                unit = job['unit'] if self.state['jobs'][jid].get('managed') else job.get('original_unit', job['unit'])
                subprocess.run(['systemctl','--user','stop','--no-block',unit],check=False)
        # Capture transition checkpoints even while all training slots are full.
        for job in self.config['jobs']:
            if job['kind'] != 'tail': continue
            state = self.state['jobs'][job['id']]
            if state['phase'] in ('needs_diagnosis','finished','running','ready','retry_wait'): continue
            root = Path(job['root']); snapshot = root / 'shared-finetune-init.pt'
            if snapshot.exists() or job.get('allow_reconstruction'):
                self.start_helper(job, 'prepare')
                continue
            parent_cfg = load_json(Path(job['parent_root'])/'config.json')
            run = Path(parent_cfg['run_dir'])
            checkpoint = run / ('finetune_init.pt' if (run / 'finetune_init.pt').exists() else 'latest.pt')
            if checkpoint.exists():
                stat = checkpoint.stat(); stamp=[stat.st_ino,stat.st_mtime_ns,stat.st_size]
                if stamp != state.get('inspected_stamp') and self.start_helper(job,'inspect'):
                    state['inspected_stamp']=stamp
        # Existing trainers, including unregistered ones, count toward the cap.
        configs = set(processes.values()) | {str(Path(j['root']) / 'config.json') for j in active_jobs}
        occupancy = len(configs) + len(processes) - len(set(processes.values()))
        helper_preparing = self.helper is not None and self.helper[2] == 'prepare'
        for job in self.config['jobs']:
            state = self.state['jobs'][job['id']]
            if (occupancy >= self.config['max_trainings']
                    or available_gib() < self.config.get('min_launch_available_gib', 6)
                    or helper_preparing): break
            if state['phase'] not in ('queued','ready','retry_wait'): continue
            if job['kind'] == 'tail' and any(
                    self.state['jobs'][p['id']]['phase'] == 'retry_wait'
                    for p in self.config['jobs'] if p['kind'] == 'parent'):
                continue
            if job['kind']=='tail' and not (Path(load_json(Path(job['root'])/'config.json')['run_dir'])/'fork.json').exists(): continue
            if state.get('next_attempt',0)>time.time(): continue
            self.launch(job)
            if state['phase']=='running': occupancy+=1
        self.state['all_finished'] = all(s['phase'] == 'finished' for s in self.state['jobs'].values())
        self.state.update(updated=datetime.datetime.now().astimezone().isoformat(),
                          active_trainings=occupancy, memory_available_gib=available_gib(),
                          helper=None if self.helper is None else dict(job=self.helper[1]['id'], mode=self.helper[2]))
        atomic_json(self.root/'state.json',self.state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    lock=(args.root/'monitor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    monitor=Monitor(args.root)
    while True:
        monitor.tick()
        time.sleep(5)


if __name__=='__main__': main()
