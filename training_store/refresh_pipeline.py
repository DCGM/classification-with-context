"""Wait for verified space preparation, refresh on NVMe, then publish an SSD copy."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .format import atomic_json, digest
from .relocate import relocate


def run(config):
    root = Path(config['work'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            preparation = Path(config['relocation_journal'])
            if (preparation / 'failure.json').exists():
                raise RuntimeError('relocation failed; inspect its failure journal before resuming')
            state = json.loads((preparation / 'status.json').read_text())
            if state['phase'] == 'complete':
                break
            atomic_json(root / 'pipeline-status.json', {'phase': 'waiting_for_verified_relocations',
                                                       'pid': os.getpid(), 'relocation': state})
            time.sleep(15)
        atomic_json(root / 'pipeline-status.json', {'phase': 'building_new_nvme_store', 'pid': os.getpid()})
        command = [sys.executable, '-u', '-m', 'training_store.refresh', '--base', config['base'],
                   '--output', config['nvme'], '--sources', config['sources'],
                   '--validation-pages', str(config.get('validation_pages', 2048))]
        subprocess.run(command, check=True)
        source, target = Path(config['nvme']), Path(config['ssd'])
        copying = target.with_name(target.name + '.copying')
        journal = root / 'new-ssd-copy.json'
        if target.exists():
            if (not journal.exists() or not json.loads(journal.read_text()).get('verified')
                    or digest(source / 'manifest.json') != digest(target / 'manifest.json')):
                raise ValueError('unexpected SSD target')
        else:
            atomic_json(root / 'pipeline-status.json', {'phase': 'copying_verified_new_store_to_ssd',
                                                       'pid': os.getpid()})
            import shutil
            required = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
            if shutil.disk_usage(target.parent).free < required + (10 << 30):
                raise ValueError('insufficient SSD headroom for new copy')
            relocate(source, copying, journal, replace_source=False)
            copying.rename(target)
        atomic_json(root / 'pipeline-status.json', {
            'phase': 'complete', 'nvme': str(source), 'ssd_new': str(target), 'ssd_old': config['base'],
            'new_manifest_sha256': digest(source / 'manifest.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        run(config)
    except BaseException as exc:
        atomic_json(Path(config['work']) / 'pipeline-failure.json',
                    {'error': str(exc), 'type': type(exc).__name__})
        raise
