"""Audit a physical-batch-only resume change, preserving the original checkpoint."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def change(root, size):
    config_file = root / 'config.json'
    config = json.loads(config_file.read_text())
    old = config['context_microbatch_size']
    if config['window_size'] != 1 or not 1 <= size <= config['batch_size'] * config['sampling_window_size']:
        raise ValueError('invalid W1 physical batch size')
    if old == size:
        return {'unchanged': True, 'physical_batch': size}
    run = Path(config['run_dir'])
    latest = run / 'latest.pt'
    backup = root / ('physical-batch-change-' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    backup.mkdir()
    shutil.copy2(config_file, backup / 'config.json')
    updated = dict(config, context_microbatch_size=size)
    record = dict(old_physical_batch=old, new_physical_batch=size,
                  backup=str(backup), checkpoint_step=None)
    if latest.exists():
        original = torch.load(latest, map_location='cpu', weights_only=False)
        if original['config'] != config:
            raise ValueError('checkpoint/config mismatch; refusing to migrate')
        shutil.copy2(latest, backup / 'latest.pt')
        migrated = dict(original, config=updated)
        temporary = latest.with_suffix('.microbatch-writing')
        torch.save(migrated, temporary)
        checked = torch.load(temporary, map_location='cpu', weights_only=False)
        assert checked['config'] == updated
        assert original.keys() == checked.keys()
        for key in original:
            if key != 'config':
                assert_same(original[key], checked[key])
        temporary.replace(latest)
        record['checkpoint_step'] = original['step']
        record['all_nonconfig_state_verified_unchanged'] = True
    temporary = config_file.with_suffix('.writing')
    temporary.write_text(json.dumps(updated, indent=2) + '\n')
    temporary.replace(config_file)
    smoke_file = root / 'smoke-config.json'
    if smoke_file.exists():
        shutil.copy2(smoke_file, backup / smoke_file.name)
        smoke = json.loads(smoke_file.read_text())
        smoke['context_microbatch_size'] = size
        smoke_file.write_text(json.dumps(smoke, indent=2) + '\n')
    (backup / 'migration.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--size', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(change(args.root, args.size), indent=2))
