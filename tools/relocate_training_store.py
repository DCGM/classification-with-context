"""Move a paused run to another copy of its native training store, preserving the original checkpoint.

Only the store location changes: the config's `index` and `training_store.path` (also in the smoke config) and
the checkpoint's recorded `provenance.training_store.path`. The run directory and ClearML task stay the same, so
training resumes into its existing history. The target must be an identical copy of the store: the same
metadata files byte for byte and the same shard names and sizes. The runner itself rehashes the catalogue at
resume and checks it against the store plan. Every other checkpoint field is verified unchanged.

Example:
    python tools/relocate_training_store.py --root /home/ikohut/tmp/context-windows-r21/T \\
        --store /home/ikohut/tmp/document-store-ocr-lines-sparse-v2
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil

import torch

from change_context_microbatch import assert_same

METADATA = ('manifest.json', 'checksums.json', 'plan.json', 'status.json', 'validation-report.json',
            'pagination-candidates-v1.json')


def shard_listing(store):
    return {p.name: p.stat().st_size for p in sorted((store / 'shards').iterdir())}


def check_identical(old, new):
    for name in METADATA:
        if (old / name).read_bytes() != (new / name).read_bytes():
            raise ValueError(f'{name} differs between the store copies')
    for name in ('catalogue.sqlite', 'documents.npy', 'pagination-candidates-v1.npy'):
        if (old / name).stat().st_size != (new / name).stat().st_size:
            raise ValueError(f'{name} sizes differ between the store copies')
    if shard_listing(old) != shard_listing(new):
        raise ValueError('shard listings differ between the store copies')


def relocate_config(config, store):
    updated = json.loads(json.dumps(config))
    updated['index'] = str(store / 'catalogue.sqlite')
    updated['training_store'] = dict(updated['training_store'], path=str(store))
    return updated


def relocate(root, store):
    store = store.resolve()
    config_file = root / 'config.json'
    config = json.loads(config_file.read_text())
    old = Path(config['training_store']['path'])
    if old.resolve() == store:
        return {'unchanged': True, 'store': str(store)}
    if Path(config['index']) != old / 'catalogue.sqlite':
        raise ValueError('index is not the catalogue of the training store')
    check_identical(old, store)
    updated = relocate_config(config, store)
    latest = Path(config['run_dir']) / 'latest.pt'
    backup = root / ('store-relocation-' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    backup.mkdir()
    shutil.copy2(config_file, backup / 'config.json')
    record = dict(old_store=str(old), new_store=str(store), backup=str(backup), checkpoint_step=None,
                  identical_files=list(METADATA), identical_shard_listing=True,
                  unchanged=['run_dir', 'clearml'])
    if latest.exists():
        original = torch.load(latest, map_location='cpu', weights_only=False)
        if original['config'] != config:
            raise ValueError('checkpoint/config mismatch; refusing to migrate')
        recorded = original['provenance']['training_store']
        if Path(recorded['path']).resolve() != old.resolve():
            raise ValueError('checkpoint provenance names another store')
        shutil.copy2(latest, backup / 'latest.pt')
        provenance = dict(original['provenance'], training_store=dict(recorded, path=str(store)))
        migrated = dict(original, config=updated, provenance=provenance)
        temporary = latest.with_suffix('.relocation-writing')
        torch.save(migrated, temporary)
        checked = torch.load(temporary, map_location='cpu', weights_only=False)
        assert checked['config'] == updated and checked['provenance'] == provenance
        assert original.keys() == checked.keys()
        for key in original:
            if key not in ('config', 'provenance'):
                assert_same(original[key], checked[key])
        for key in original['provenance']:
            if key != 'training_store':
                assert_same(original['provenance'][key], checked['provenance'][key])
        temporary.replace(latest)
        record.update(checkpoint_step=original['step'], all_other_state_verified_unchanged=True)
    temporary = config_file.with_suffix('.writing')
    temporary.write_text(json.dumps(updated, indent=2) + '\n')
    temporary.replace(config_file)
    smoke_file = root / 'smoke-config.json'
    if smoke_file.exists():
        shutil.copy2(smoke_file, backup / smoke_file.name)
        smoke_file.write_text(json.dumps(relocate_config(json.loads(smoke_file.read_text()), store), indent=2) + '\n')
    (backup / 'migration.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='Recipe directory with config.json')
    parser.add_argument('--store', type=Path, required=True, help='The other copy of the native store')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(relocate(args.root, args.store), indent=2))
