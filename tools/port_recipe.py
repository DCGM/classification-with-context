"""Port a local training recipe to a remote machine: the same experiment, only machine paths changed.

The recipe (config.json, smoke-config.json, smoke.py, launch.py, smoke-dev.jsonl, expected-initialization.json)
is copied with the store, catalogue, dev manifest, pagination sidecar, run directory and Hugging Face cache
pointing to the remote copies, the disk-slot part of the run name replaced by the machine, and the source frozen
from a committed revision (`git archive`). The vector LMDB paths are kept: a published store records them and
training never opens them. Optionally the image source is replaced (with a store built for it by
`training_store.refresh_images`). `run.sh` supervises the job: preflight, training, resume after failures.

Example:
    python tools/port_recipe.py --recipe /home/ikohut/tmp/window-series-r22/w64 --machine sophie1 \\
        --output /home/ikohut/tmp/ported-recipes/window-series-r22/w64
"""
import argparse
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN = ['page_sequence', 'training_store', 'embeddings', 'pyproject.toml']
HOME = '/homes/ikohut'
DATA = f'{HOME}/data/classification-with-context'
EXPERIMENTS = f'{HOME}/experiments/classification-with-context'
STORE = f'{DATA}/document-store-ocr-lines-sparse-v2'

RUN_SH = r'''#!/bin/bash
# Supervise the job on one GPU: preflight + training via launch.py, resume after failures (at most 5 attempts).
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=~/python_env/classification-with-context/bin/python
RUN=$(python3 -c "import json;print(json.load(open('config.json'))['run_dir'])")
for attempt in 1 2 3 4 5; do
  args=(); [ -e "$RUN/latest.pt" ] && args=(--resume)
  echo "{\"time\": \"$(date -Is)\", \"attempt\": $attempt, \"resume\": ${#args[@]}, \"gpu\": \"$CUDA_VISIBLE_DEVICES\"}" >> attempts.jsonl
  $PY -u launch.py "${args[@]}" >> managed-training.log 2>&1 && { echo "{\"time\": \"$(date -Is)\", \"finished\": true}" >> attempts.jsonl; exit 0; }
  code=$?
  echo "{\"time\": \"$(date -Is)\", \"attempt\": $attempt, \"exit_code\": $code}" >> attempts.jsonl
  [ -e smoke-failed.json ] && exit 1
  [ -e "$RUN/latest.pt" ] || [ -e smoke-passed.json ] || exit 1
  sleep 120
done
exit 1
'''


def run_name(name, machine, tag=None):
    name, count = re.subn(r'^(bert-mini-fp16-)(nvme|ssd)-', rf'\g<1>{machine}-', name)
    if count != 1:
        raise ValueError('run name without a disk slot')
    if tag:
        name = name.replace('-scattered-', f'-scattered-{tag}-', 1)
    return name


def adapt(config, root, machine, store, image=None, tag=None, smoke=False):
    config = json.loads(json.dumps(config))
    config['index'] = f'{store}/catalogue.sqlite'
    config['training_store'] = dict(config['training_store'], path=store)
    if image:
        config['features']['sources']['image']['stores'] = dict(image)
    if config.get('pagination'):
        config['pagination'] = dict(config['pagination'],
                                    path=f'{DATA}/pagination-experiment-w128/pagination.npy')
    if smoke:
        config['run_dir'] = f'{root}/smoke-run'
        config['dev_manifest'] = f'{root}/smoke-dev.jsonl'
    else:
        name = run_name(Path(config['run_dir']).name, machine, tag)
        config['run_dir'] = f'{EXPERIMENTS}/page-sequence-runs/{name}'
        config['dev_manifest'] = f'{DATA}/page-sequence-data/dev.jsonl'
        config['clearml'] = dict(config['clearml'], name=name)
    return config


def port(recipe, output, root, machine, revision='HEAD', store=STORE, image=None, tag=None):
    if output.exists():
        raise FileExistsError(output)
    if subprocess.run(['git', 'status', '--porcelain', '--', *FROZEN], cwd=REPOSITORY,
                      capture_output=True, text=True, check=True).stdout.strip() and revision == 'HEAD':
        raise RuntimeError('frozen sources have uncommitted changes; commit before porting')
    commit = subprocess.run(['git', 'rev-parse', revision], cwd=REPOSITORY, capture_output=True, text=True,
                            check=True).stdout.strip()
    output.mkdir(parents=True)
    archive = subprocess.run(['git', 'archive', commit, *FROZEN], cwd=REPOSITORY, capture_output=True,
                             check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(output / 'code', filter='data')
    local = json.loads((recipe / 'config.json').read_text())
    config = adapt(local, root, machine, store, image, tag)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    smoke = adapt(json.loads((recipe / 'smoke-config.json').read_text()), root, machine, store, image, tag, True)
    (output / 'smoke-config.json').write_text(json.dumps(smoke, indent=2) + '\n')
    diff = {k: {'before': local.get(k), 'after': v} for k, v in config.items() if local.get(k) != v}
    (output / 'config-diff.json').write_text(json.dumps(
        diff | {'derived_from': str(recipe), 'source_commit': commit, 'machine': machine}, indent=2) + '\n')
    for name in ('smoke-dev.jsonl', 'expected-initialization.json', 'smoke.py'):
        shutil.copy2(recipe / name, output / name)
    launch = (recipe / 'launch.py').read_text()
    if launch.count("'/home/ikohut/tmp/huggingface'") != 1:
        raise ValueError('unexpected launcher')
    (output / 'launch.py').write_text(launch.replace("'/home/ikohut/tmp/huggingface'",
                                                     f"'{EXPERIMENTS}/huggingface'"))
    (output / 'run.sh').write_text(RUN_SH)
    (output / 'run.sh').chmod(0o755)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recipe', type=Path, required=True, help='Local recipe directory')
    parser.add_argument('--output', type=Path, required=True, help='Local staging directory (copied afterwards)')
    parser.add_argument('--remote-root', help='Recipe directory on the machine (default: mirrors the series)')
    parser.add_argument('--machine', required=True)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--store', default=STORE)
    parser.add_argument('--image-public')
    parser.add_argument('--image-private')
    parser.add_argument('--tag', help='Inserted into the run name, e.g. dinoorig')
    args = parser.parse_args()
    root = args.remote_root or f'{EXPERIMENTS}/{args.recipe.parent.name}/{args.recipe.name}'
    image = ({'public': args.image_public, 'private': args.image_private}
             if args.image_public or args.image_private else None)
    config = port(args.recipe, args.output, root, args.machine, args.revision, args.store, image, args.tag)
    print(json.dumps({'root': root, 'run_dir': config['run_dir']}, indent=2))
