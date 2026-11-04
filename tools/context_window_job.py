"""Frozen job entry point: verify real-data CUDA preflight before full training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ | {'PYTHONPATH': str(root / 'code'), 'HF_HOME': '/home/ikohut/tmp/huggingface',
                        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    if not (root / 'smoke-passed.json').exists():
        if (root / 'smoke-failed.json').exists():
            raise RuntimeError('preflight verification failed; diagnose before retrying')
        with (root / 'smoke.log').open('ab', buffering=0) as log:
            result = subprocess.run([sys.executable, '-u', str(root / 'smoke.py')],
                                    cwd=root / 'code', env=env, stdout=log, stderr=log)
        if result.returncode:
            (root / 'smoke-failed.json').write_text(json.dumps({'exit_code': result.returncode}) + '\n')
            raise RuntimeError('preflight verification failed; inspect smoke.log')
    command = [sys.executable, '-u', '-m', 'page_sequence.runner', 'train',
               '--config', str(root / 'config.json')]
    config = json.loads((root / 'config.json').read_text())
    if args.resume:
        command += ['--resume', str(Path(config['run_dir']) / 'latest.pt')]
        env['CLEARML_SET_ITERATION_OFFSET'] = '0'
    os.chdir(root / 'code')
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    main()
