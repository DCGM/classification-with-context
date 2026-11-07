"""Change a stopped run's phase budgets (`pretrain_steps`, `finetune_steps`), preserving the original checkpoint.

Only the config's budget fields change, in `config.json` and in the resume checkpoint `latest.pt`. Every other
checkpoint field is verified unchanged. A budget may not be set below the progress already made. Phase snapshots
(`initial.pt`, ...) keep the configuration they were written with. Originals go to `<root>/budget-change-<time>/`.

Example:
    python tools/change_phase_budget.py --root /home/ikohut/tmp/window-series-r22/w512 --pretrain-steps 16000
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil

import torch

from change_context_microbatch import assert_same


def change(root, budgets):
    config_file = root / 'config.json'
    config = json.loads(config_file.read_text())
    updated = dict(config, **budgets)
    if updated == config:
        return {'unchanged': True}
    backup = root / ('budget-change-' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    backup.mkdir()
    shutil.copy2(config_file, backup / 'config.json')
    record = dict(before={k: config[k] for k in budgets}, after=budgets, backup=str(backup), checkpoint_step=None)
    latest = Path(config['run_dir']) / 'latest.pt'
    if latest.exists():
        original = torch.load(latest, map_location='cpu', weights_only=False)
        if original['config'] != config:
            raise ValueError('checkpoint/config mismatch; refusing to migrate')
        if original['step'] > updated['pretrain_steps'] + updated['finetune_steps'] or (
                original['training']['stage'] == 'pretrain' and original['step'] > updated['pretrain_steps']):
            raise ValueError('the new budget is below the progress already made')
        shutil.copy2(latest, backup / 'latest.pt')
        migrated = dict(original, config=updated)
        temporary = latest.with_suffix('.budget-writing')
        torch.save(migrated, temporary)
        checked = torch.load(temporary, map_location='cpu', weights_only=False)
        assert checked['config'] == updated and original.keys() == checked.keys()
        for key in original:
            if key != 'config':
                assert_same(original[key], checked[key])
        temporary.replace(latest)
        record.update(checkpoint_step=original['step'], all_other_state_verified_unchanged=True)
    temporary = config_file.with_suffix('.writing')
    temporary.write_text(json.dumps(updated, indent=2) + '\n')
    temporary.replace(config_file)
    (backup / 'migration.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='Recipe directory with config.json')
    parser.add_argument('--pretrain-steps', type=int)
    parser.add_argument('--finetune-steps', type=int)
    args = parser.parse_args()
    budgets = {k: v for k, v in (('pretrain_steps', args.pretrain_steps), ('finetune_steps', args.finetune_steps))
               if v is not None}
    torch.set_num_threads(1)
    print(json.dumps(change(args.root, budgets), indent=2))
