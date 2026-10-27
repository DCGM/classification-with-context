"""Lightweight slot gate: preserve prepared fork while another SSD run finishes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time


def wait_for_slot(root, status_path, poll_seconds=3):
    while True:
        if status_path.exists():
            report = json.loads(status_path.read_text())
            jobs = report['jobs']
            if not jobs:
                raise ValueError('blocking status has no jobs')
            if any(job['state'] == 'failed' for job in jobs.values()):
                raise RuntimeError('SSD occupant failed; inspect before releasing slot')
            if all(job['state'] == 'finished' and job['exit_code'] == 0 for job in jobs.values()):
                return
        temporary = root / 'pipeline-status.writing'
        temporary.write_text(json.dumps({'updated': datetime.datetime.now().astimezone().isoformat(),
                                         'phase': 'waiting_for_ssd_slot',
                                         'blocking_status': str(status_path)}, indent=2) + '\n')
        temporary.replace(root / 'pipeline-status.json')
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--status-path', type=Path, required=True)
    args = parser.parse_args()
    try:
        wait_for_slot(args.root, args.status_path)
        os.chdir(args.root)
        os.execve(sys.executable, [sys.executable, '-u', str(args.root / 'supervise.py')], os.environ)
    except BaseException as error:
        (args.root / 'pipeline-status.json').write_text(json.dumps({'phase': 'failed', 'error': str(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
