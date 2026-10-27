"""Bounded checkpoint inspection and audited alternate fine-tuning initialization.

Run with the parent's frozen source on PYTHONPATH, never the live checkout.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import torch


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def inspect(parent, target, output):
    from fork_finetune import require_transition
    config = json.loads((parent / 'config.json').read_text())
    source = Path(config['run_dir']) / 'latest.pt'
    candidate = target / 'inspect-candidate.pt'
    # Opening the source survives the trainer's atomic checkpoint replacement.
    shutil.copyfile(source, candidate)
    state = torch.load(candidate, map_location='cpu', weights_only=False)
    training = state.get('training') or {}
    snapshot = target / 'shared-finetune-init.pt'
    result = dict(step=state['step'], training=training, snapshot=snapshot.exists())
    if training.get('stage') == 'finetune' and training.get('stage_step') == 0:
        require_transition(state)
        if not snapshot.exists():
            candidate.replace(snapshot)
        result['snapshot'] = True
        result['exact_transition'] = True
    candidate.unlink(missing_ok=True)
    atomic_json(output, result)


def reconstruct(parent, target):
    """Only for an already finished parent whose transition was not retained."""
    from fork_finetune import file_sha256
    config = json.loads((parent / 'config.json').read_text())
    run = Path(config['run_dir'])
    latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
    if not latest['training'].get('finished'):
        raise ValueError('reconstruction requires an already completed parent')
    # Global step is the actual transition step, not the earlier best-model step.
    transition_step = latest['step'] - latest['training']['stage_step']
    del latest
    source = run / 'pretrain_best.pt'
    state = torch.load(source, map_location='cpu', weights_only=False)
    if state['training']['stage'] != 'pretrain' or state['training'].get('pending_validation'):
        raise ValueError('invalid best pretrained checkpoint')
    original_step = state['step']
    state['step'] = transition_step
    state['training'] = dict(stage='finetune', stage_step=0, stage_best=-1.,
                             bad_validations=0, finished=False)
    state['sampler']['stage'] = 'finetune'
    state['sampler']['pending_sources'] = []
    snapshot = target / 'shared-finetune-init.pt'
    temporary = snapshot.with_suffix('.writing')
    torch.save(state, temporary)
    os.replace(temporary, snapshot)
    atomic_json(target / 'transition-reconstruction.json', {
        'exact_parent_transition': False,
        'reason': 'completed parent did not retain fine-tuning step-zero checkpoint',
        'source': str(source), 'source_sha256': file_sha256(source),
        'source_step': original_step, 'global_transition_step': transition_step,
        'preserved': ['model', 'optimizer', 'grad_scaler', 'features', 'config', 'provenance'],
        'limitation': 'sampler and RNG come from best pretraining checkpoint; not the original tail continuation'})
    return snapshot


def prepare(parent, target, output, allow_reconstruction=False):
    from fork_finetune import create_fork
    config = json.loads((target / 'config.json').read_text())
    run = Path(config['run_dir'])
    if (run / 'fork.json').exists() and (run / 'fork-init.pt').exists() and (run / 'clearml.json').exists():
        atomic_json(output, {'prepared': True, 'checkpoint': str(run / 'fork-init.pt')})
        return
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('partial fork preparation requires diagnosis; no overwrite')
    snapshot = target / 'shared-finetune-init.pt'
    if not snapshot.exists():
        if not allow_reconstruction:
            raise ValueError('exact transition snapshot not captured')
        snapshot = reconstruct(parent, target)
    result = create_fork(snapshot, config)
    atomic_json(output, {'prepared': True, 'checkpoint': str(result),
                        'exact_transition': not (target / 'transition-reconstruction.json').exists()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'prepare'])
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-reconstruction', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'inspect':
        inspect(args.parent, args.target, args.output)
    else:
        prepare(args.parent, args.target, args.output, args.allow_reconstruction)


if __name__ == '__main__':
    main()
