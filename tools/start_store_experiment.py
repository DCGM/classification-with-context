"""Wait for a published native store, smoke-test a frozen recipe, then monitor it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def store_ready(store, failures):
    for path in failures:
        if path.exists():
            raise RuntimeError(f'Data preparation failed; inspect {path}')
    manifest = store / 'manifest.json'
    status = store / 'status.json'
    if not manifest.exists() or not status.exists():
        return False
    info = json.loads(manifest.read_text())
    return info.get('complete') is True and json.loads(status.read_text()).get('phase') == 'complete'


def run(root):
    settings = json.loads((root / 'start-config.json').read_text())
    status = root / 'start-status.json'
    store = Path(settings['store'])
    while not store_ready(store, [Path(p) for p in settings['failure_journals']]):
        write(status, dict(phase='waiting_for_published_store', store=str(store), updated=time.time()))
        time.sleep(15)
    python = settings['python']
    passed = root / 'smoke-passed.json'
    if not passed.exists():
        from training_monitor import available_gib, training_processes
        limit = json.loads((root / 'monitor-config.json').read_text())['max_trainings']
        required = json.loads((root / 'monitor-config.json').read_text()).get('min_launch_available_gib', 6)
        while available_gib() < required or len(training_processes()) >= limit:
            write(status, dict(phase='waiting_for_smoke_slot', updated=time.time()))
            time.sleep(15)
        write(status, dict(phase='smoke_testing', updated=time.time()))
        with (root / 'smoke.log').open('ab', buffering=0) as log:
            subprocess.run([python, '-u', str(root / 'launch.py'), '--smoke'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([python, str(root / 'verify-smoke.py')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        write(passed, dict(passed=True, updated=time.time()))
    write(status, dict(phase='training_monitor', updated=time.time()))
    os.execv(sys.executable, [sys.executable, '-u', str(root / 'training_monitor.py'), '--root', str(root)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.root)
    except BaseException as exc:
        write(args.root / 'start-failure.json', dict(error=str(exc), type=type(exc).__name__))
        raise
