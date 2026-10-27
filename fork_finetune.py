"""Audited fine-tuning fork; run with the parent's immutable training code.

The input must be the actual fine-tuning step-zero checkpoint. Native resume
identity checks are preserved; only the explicitly checked experiment changes
and equivalent local-store paths are published into a new checkpoint/config.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from page_sequence.checkpoint_migration import protected_digest
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.losses import stage_loss_weights
from page_sequence.runner import make_dataset, provenance
from page_sequence.schema import PAGE_TYPES, PLACEMENTS
from page_sequence.tracking import Tracking


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_transition(state):
    training = state.get('training') or {}
    if (state.get('format') != 1 or tuple(state['page_types']) != PAGE_TYPES
            or tuple(state['placements']) != PLACEMENTS
            or training.get('stage') != 'finetune' or training.get('stage_step') != 0
            or training.get('finished') or training.get('pending_validation')
            or state['sampler']['stage'] != 'finetune' or state['sampler']['pending_sources']):
        raise ValueError('fork requires the actual fine-tuning step-zero transition checkpoint')


def check_config(parent, target):
    allowed = {'index', 'run_dir', 'clearml', 'training_store', 'pagination', 'stage_loss_weights'}
    for key in parent.keys() | target.keys():
        if key not in allowed and parent.get(key) != target.get(key):
            raise ValueError(f'unsupported fine-tuning fork change: {key}')
    for key, path_key in [('clearml', 'name'), ('training_store', 'path'), ('pagination', 'path')]:
        a = {k: v for k, v in parent.get(key, {}).items() if k != path_key}
        b = {k: v for k, v in target.get(key, {}).items() if k != path_key}
        if a != b:
            raise ValueError(f'unsupported fine-tuning fork change: {key}')
    if stage_loss_weights(parent, 'pretrain') != stage_loss_weights(target, 'pretrain'):
        raise ValueError('fork must preserve pretraining weights')
    before = stage_loss_weights(parent, 'finetune')
    after = stage_loss_weights(target, 'finetune')
    expected = replace(before, weak_type=.05)
    if before.weak_type != .2 or after != expected:
        raise ValueError('only fine-tuning weak_type 0.2 -> 0.05 is authorized by this fork')
    if Path(parent['run_dir']).resolve() == Path(target['run_dir']).resolve():
        raise ValueError('fork requires a distinct run directory')


def check_provenance(parent, target):
    normalized = copy.deepcopy(target)
    for key in ('training_store', 'pagination'):
        if key in normalized and key in parent:
            normalized[key]['path'] = parent[key]['path']
    if normalized != parent:
        raise ValueError('fork data/code/environment/feature provenance differs from parent')


def create_fork(checkpoint, config):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require_transition(state)
    original_config, original_prov = state['config'], state['provenance']
    check_config(original_config, config)
    run_dir = Path(config['run_dir'])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError('refusing to overwrite an existing fork run')
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = make_dataset(index, features, config)
    target_prov = provenance(index, config['dev_manifest'])
    target_prov['vector_sources'] = features.source_provenance()
    if config.get('training_store'):
        target_prov['training_store'] = dataset.source_provenance()
        if index.fingerprint() != dataset.plan['index_sha256']:
            raise ValueError('fork catalogue does not match native store')
    if config.get('pagination'):
        target_prov['pagination'] = dataset.pagination_provenance()
    check_provenance(original_prov, target_prov)
    if features.state_dict() != state['features']:
        raise ValueError('fork feature contract changed')
    protected = protected_digest(state)
    original_sampler = state['sampler']
    audit = {'parent_checkpoint': str(checkpoint.resolve()), 'parent_checkpoint_sha256': file_sha256(checkpoint),
             'shared_global_step': state['step'], 'shared_finetune_step': 0,
             'protected_training_state_sha256': protected, 'parent_run_dir': original_config['run_dir'],
             'parent_source_sha256': original_prov['code_sha256'],
             'fork_source_sha256': target_prov['code_sha256'], 'helper_sha256': file_sha256(__file__),
             'change': 'fine-tuning weak_type coefficient 0.2 -> 0.05; equivalent SSD store paths'}
    state['config'], state['provenance'] = config, target_prov
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / 'fork-init.writing'
    torch.save(state, temporary)
    checked = torch.load(temporary, map_location='cpu', weights_only=False)
    normalized = dict(checked, config=original_config, provenance=original_prov)
    if protected_digest(normalized) != protected or checked['sampler'] != original_sampler:
        raise ValueError('fork changed model/optimizer/sampler/RNG/scaler/progress')
    os.replace(temporary, run_dir / 'fork-init.pt')
    # Separate task/history; the trainer will resume this explicit fork checkpoint.
    tracking = Tracking(config, target_prov, resume=False)
    if tracking.task is not None:
        tracking.task.connect_configuration(audit, name='shared_pretraining_fork')
    tracking.close()
    (run_dir / 'fork.json').write_text(json.dumps(audit, indent=2) + '\n')
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return run_dir / 'fork-init.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = create_fork(args.checkpoint, json.loads(Path(args.config).read_text()))
    print(json.dumps({'fork_checkpoint': str(path)}), flush=True)


if __name__ == '__main__':
    main()
