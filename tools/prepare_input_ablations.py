"""Freeze the r23 input ablations: the r22 W512 recipe without the optional signals, cumulatively.

- A `no-edges-metadata`: no page-start/page-end auxiliary heads (and no losses for them), and no document
  start/end metadata on the input (the five-bit features keep only the scan start/end flags);
- B `no-scan-flags`: A without the scan start/end flags as well;
- C `no-page-numbers`: B without the page numbers as well, i.e. only the page embeddings on the input.

The signals are removed inside the model (`models.DISABLEABLE_INPUTS`: zeroed in its forward pass for training,
validation and every evaluation), so the sampled data, the augmentation randomness and the initialization of every
remaining parameter are those of r22 W512; only the four auxiliary head tensors are missing. The expected initial
parameters are therefore the r22 W512 initialization (`initial.pt` of its run) without those tensors.
Evaluations with provided document metadata are dropped (no model is trained with it).

Example:
    python tools/prepare_input_ablations.py --reference /home/ikohut/tmp/window-series-r22/w512 \\
        --reference-run /home/ikohut/tmp/page-sequence-runs/<r22 W512 run> --output /home/ikohut/tmp/input-ablations-r23
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
AUXILIARY = ('page_start_head.weight', 'page_start_head.bias', 'page_end_head.weight', 'page_end_head.bias')
VARIANTS = [('a-no-edges-metadata', 'noedgeheads-nometadata', ['document_metadata']),
            ('b-no-scan-flags', 'noedgeheads-nometadata-noscanflags', ['document_metadata', 'scan_flags']),
            ('c-no-page-numbers', 'noedgeheads-nometadata-noscanflags-nopagenumbers',
             ['document_metadata', 'scan_flags', 'page_numbers'])]


def digest(state, skip=()):
    h = hashlib.sha256()
    for name, tensor in state.items():
        if name not in skip:
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def variant(config, tag, disabled, smoke=False):
    config = json.loads(json.dumps(config))
    config.update(auxiliary_edges=False, disabled_inputs=disabled, document_metadata_evaluation=False)
    config['loss_weights'] = dict(config['loss_weights'], page_start=0.0, page_end=0.0)
    if not smoke:
        name = Path(config['run_dir']).name
        if name.count('-boundary5-metadata50-') != 1 or not name.endswith('-r22'):
            raise ValueError('unexpected reference run name')
        name = name.replace('-boundary5-metadata50-', f'-{tag}-').removesuffix('-r22') + '-r23'
        config['run_dir'] = str(Path(config['run_dir']).parent / name)
        config['clearml'] = dict(config['clearml'], name=name)
    return config


def prepare(reference, reference_run, output):
    if output.exists():
        raise FileExistsError(output)
    expected = json.loads((reference / 'expected-initialization.json').read_text())['sha256']
    initial = torch.load(reference_run / 'initial.pt', map_location='cpu', weights_only=False)['model']
    if digest(initial) != expected:
        raise ValueError('the reference run does not start from the expected initialization')
    if not set(AUXILIARY) <= set(initial):
        raise ValueError('the reference has no auxiliary heads')
    without_heads = digest(initial, AUXILIARY)
    source = json.loads((reference / 'config.json').read_text())
    smoke = json.loads((reference / 'smoke-config.json').read_text())
    jobs = []
    for directory, tag, disabled in VARIANTS:
        root = output / directory
        root.mkdir(parents=True)
        config = variant(source, tag, disabled)
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (root / 'config-diff.json').write_text(json.dumps(
            {k: {'before': source.get(k), 'after': v} for k, v in config.items() if source.get(k) != v}
            | {'derived_from': str(reference)}, indent=2) + '\n')
        test = variant(smoke, tag, disabled, smoke=True)
        test.update(run_dir=str(root / 'smoke-run'), dev_manifest=str(root / 'smoke-dev.jsonl'))
        (root / 'smoke-config.json').write_text(json.dumps(test, indent=2) + '\n')
        (root / 'expected-initialization.json').write_text(json.dumps(
            {'sha256': without_heads, 'derived_from': f'{reference_run}/initial.pt without {list(AUXILIARY)}'},
            indent=2) + '\n')
        shutil.copy2(reference / 'smoke-dev.jsonl', root / 'smoke-dev.jsonl')
        shutil.copy2(reference / 'launch.py', root / 'launch.py')
        shutil.copy2(REPOSITORY / 'tools/window_series_smoke.py', root / 'smoke.py')
        jobs.append(dict(id=f'r23-{directory}', root=str(root), run_dir=config['run_dir'], disabled_inputs=disabled))
    (output / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True, help='r22 W512 recipe directory')
    parser.add_argument('--reference-run', type=Path, required=True, help='Its run directory (for initial.pt)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference, args.reference_run, args.output), indent=2))
