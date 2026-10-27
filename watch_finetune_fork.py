"""Capture the parent's atomic fine-tuning initialization and launch one audited fork."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from fork_finetune import require_transition


def write_status(root, phase, **fields):
    temporary = root / 'pipeline-status.writing'
    temporary.write_text(json.dumps({'updated': datetime.datetime.now().astimezone().isoformat(),
                                     'phase': phase, **fields}, indent=2) + '\n')
    temporary.replace(root / 'pipeline-status.json')


def capture_transition(parent_root, target_root, poll_seconds=2):
    config = json.loads((parent_root / 'config.json').read_text())
    run_dir = Path(config['run_dir'])
    source = run_dir / 'latest.pt'
    snapshot = target_root / 'shared-finetune-init.pt'
    if snapshot.exists():
        raise FileExistsError('explicit recovery required for an existing shared snapshot')
    last_stamp = None
    while True:
        status = json.loads((parent_root / 'status.json').read_text())
        if any(job['state'] == 'failed' for job in status['jobs'].values()):
            raise RuntimeError('parent run stopped unsuccessfully; inspect before forking')
        source = (run_dir / 'finetune_init.pt' if (run_dir / 'finetune_init.pt').exists()
                  else run_dir / 'latest.pt')
        if source.exists():
            stat = source.stat()
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if stamp != last_stamp:
                candidate = target_root / 'snapshot-candidate.pt'
                shutil.copyfile(source, candidate)  # open descriptor survives atomic parent replacement
                state = torch.load(candidate, map_location='cpu', weights_only=False)
                phase = (state.get('training') or {}).get('stage')
                if phase == 'finetune':
                    require_transition(state)  # never substitute a later fine-tuned checkpoint
                    step = state['step']
                    del state
                    candidate.replace(snapshot)
                    write_status(target_root, 'captured_transition', shared_global_step=step,
                                 snapshot=str(snapshot))
                    return snapshot
                if phase != 'pretrain':
                    raise ValueError('unknown parent checkpoint stage')
                del state
                last_stamp = stamp
        if all(job['state'] == 'finished' for job in status['jobs'].values()):
            raise RuntimeError('parent finished before a fine-tuning initialization was captured')
        write_status(target_root, 'waiting_for_nvme_pretraining', parent_run_dir=config['run_dir'])
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for-status', type=Path,
                        help='Capture and prepare first, then wait for this SSD occupant to finish')
    args = parser.parse_args()
    root = args.target_root.resolve()
    try:
        snapshot = capture_transition(args.parent_root.resolve(), root)
        env = os.environ | {'PYTHONPATH': str(root / 'code'), 'HF_HOME': '/home/ikohut/tmp/huggingface',
                            'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                            'CLEARML_SET_ITERATION_OFFSET': '0'}
        write_status(root, 'preparing_audited_fork', snapshot=str(snapshot))
        with (root / 'fork.log').open('ab', buffering=0) as output:
            subprocess.run([sys.executable, str(root / 'fork_finetune.py'), '--checkpoint', str(snapshot),
                            '--config', str(root / 'config.json')], cwd=root / 'code', env=env,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        write_status(root, 'fork_prepared', snapshot=str(snapshot))
        if not args.prepare_only:
            # Replace this Torch-loaded watcher with the small supervisor, freeing
            # its runtime/model-checkpoint memory before the second trainer starts.
            os.chdir(root)
            if args.wait_for_status:
                os.execve(sys.executable, [sys.executable, '-u', str(root / 'wait_training_slot.py'),
                          '--root', str(root), '--status-path', str(args.wait_for_status.resolve())], env)
            os.execve(sys.executable, [sys.executable, '-u', str(root / 'supervise.py')], env)
    except BaseException as error:
        write_status(root, 'failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
