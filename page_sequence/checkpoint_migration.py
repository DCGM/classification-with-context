"""Explicit offline migration of the known legacy pair-history memory bug.

Original checkpoints are preserved. Only diagnostic sampler state and the audited
source digest change; config/data/environment/model/optimizer/RNG checks stay strict.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .cardinality import PairCardinality

LEGACY_SOURCE = 'e88807379a3eee335af520966685c338f654f69924c390ff0149cd48230faabb'
CHANGED_FILES = {'page_sequence/cardinality.py', 'page_sequence/checkpoint_migration.py',
                 'page_sequence/sampling.py', 'page_sequence/runner.py'}


def source_files(root):
    root = Path(root)
    return (sorted((root / 'page_sequence').glob('*.py'))
            + sorted((root / 'training_store').glob('*.py'))
            + [root / 'embeddings/text/page_stats/page_stats.py'])


def source_digest(root):
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def protected_digest(state):
    """Content identity for training state, excluding diagnostic bookkeeping."""
    sampler = {k: v for k, v in state['sampler'].items()
               if k not in ('seen_pairs', 'pair_cardinality', 'counters')}
    sampler['counters'] = {k: v for k, v in state['sampler']['counters'].items()
                           if k not in ('unique_join_pairs', 'unique_join_pairs_estimate')}
    protected = {k: v for k, v in state.items() if k not in ('sampler', 'provenance', 'migration_history')}
    protected.update(sampler=sampler, provenance={k: v for k, v in state['provenance'].items()
                                                 if k != 'code_sha256'})
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            digest.update(b'dict')
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        elif isinstance(value, (set, frozenset)):
            digest.update(type(value).__name__.encode())
            for item in sorted(value, key=repr):
                visit(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b'\0')
    visit(protected)
    return digest.hexdigest()


def compact_sampler(state):
    legacy = state['sampler']
    counter = PairCardinality()
    for pair in legacy['seen_pairs']:
        counter.add(pair)
    compact = {k: v for k, v in legacy.items() if k != 'seen_pairs'}
    compact['pair_cardinality'] = counter.state_dict()
    compact['counters'] = dict(legacy['counters'])
    exact = compact['counters'].pop('unique_join_pairs', len(legacy['seen_pairs']))
    compact['counters']['unique_join_pairs_estimate'] = counter.estimate()
    state['sampler'] = compact
    return {'old_exact_pairs': exact, 'new_pair_estimate': counter.estimate(),
            'register_bytes': counter.registers.nbytes}


def migrate_run(run_dir, old_code, new_code):
    run_dir, old_code, new_code = map(Path, (run_dir, old_code, new_code))
    old_digest, new_digest = source_digest(old_code), source_digest(new_code)
    if old_digest != LEGACY_SOURCE:
        raise ValueError('migration supports only the audited legacy source version')
    inventories = [{str(p.relative_to(root)): file_digest(p) for p in source_files(root)}
                   for root in (old_code, new_code)]
    changed = {k for k in inventories[0].keys() | inventories[1].keys()
               if inventories[0].get(k) != inventories[1].get(k)}
    if changed != CHANGED_FILES:
        raise ValueError(f'unexpected source changes: {sorted(changed)}')
    config = json.loads((run_dir / 'config.json').read_text())
    backup = run_dir / 'memory-fix-originals'
    backup.mkdir(exist_ok=False)
    report = {'old_source_sha256': old_digest, 'new_source_sha256': new_digest,
              'changed_files': sorted(changed), 'checkpoints': {}}
    for name in ('latest.pt', 'best.pt', 'pretrain_best.pt', 'finetune_best.pt'):
        path = run_dir / name
        if not path.exists():
            continue
        shutil.copy2(path, backup / name)
        original_sha = file_digest(path)
        if file_digest(backup / name) != original_sha:
            raise ValueError('backup checksum mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['config'] != config or state['provenance']['code_sha256'] != old_digest:
            raise ValueError('checkpoint config/source mismatch')
        for package, version in state['provenance']['versions'].items():
            if importlib.metadata.version(package) != version:
                raise ValueError('checkpoint environment mismatch')
        identity = protected_digest(state)
        counts = compact_sampler(state)
        state['provenance'] = dict(state['provenance'], code_sha256=new_digest)
        audit = dict(counts, old_checkpoint_sha256=original_sha, protected_state_sha256=identity,
                     old_source_sha256=old_digest, new_source_sha256=new_digest, step=state['step'])
        state['migration_history'] = state.get('migration_history', []) + [audit]
        temporary = path.with_suffix('.memoryfix.writing')
        torch.save(state, temporary)
        del state
        checked = torch.load(temporary, map_location='cpu', weights_only=False)
        if protected_digest(checked) != identity:
            raise ValueError('migration changed protected training state')
        del checked
        os.replace(temporary, path)
        report['checkpoints'][name] = dict(audit, new_checkpoint_sha256=file_digest(path))
        (backup / 'migration.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'checkpoint': str(path), **audit}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--old-code', required=True)
    parser.add_argument('--new-code', required=True)
    args = parser.parse_args()
    migrate_run(args.run_dir, args.old_code, args.new_code)


if __name__ == '__main__':
    main()
