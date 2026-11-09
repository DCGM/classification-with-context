"""Freeze the r23 input ablations: the r22 W512 recipe without the optional signals, cumulatively.

- A `no-edges-metadata`: no page-start/page-end auxiliary heads (and no losses for them), and no document
  start/end metadata on the input (the five-bit features keep only the scan start/end flags);
- B `no-scan-flags`: A without the scan start/end flags as well;
- C `no-page-numbers`: B without the page numbers as well, i.e. only the page embeddings on the input.

The remaining combinations of the three input signals (page numbers, scan start/end flags, document start/end
metadata); a model given the document metadata keeps the page-start/page-end heads of r22 W512:
- D `scan-flags-only`: only the scan start/end flags;
- E `metadata-only`: only the document start/end metadata (with the heads);
- F `metadata-scan-flags`: the document metadata and the scan flags (with the heads), no page numbers;
- G `metadata-page-numbers`: the document metadata and the page numbers (with the heads), no scan flags.

The models given the document metadata once more without the page-start/page-end heads:
- H `metadata-only-noheads`, I `metadata-scan-flags-noheads`, J `metadata-page-numbers-noheads`: E, F, G without
  the heads;
- K `all-signals-noheads`: r22 W512 without the heads.

The signals are removed inside the model (`models.DISABLEABLE_INPUTS`: zeroed in its forward pass for training,
validation and every evaluation), so the sampled data, the augmentation randomness and the initialization of every
remaining parameter are those of r22 W512; without the heads only the four auxiliary head tensors are missing. The
expected initial parameters are therefore the r22 W512 initialization (`initial.pt` of its run), without those
tensors for the models without the heads. Evaluations with provided document metadata are kept only for the models
given it. `--variants` prepares a selection into an existing output directory, extending its `jobs.json`.

Example:
    python tools/prepare_input_ablations.py --reference /home/ikohut/tmp/window-series-r22/w512 \\
        --reference-run /home/ikohut/tmp/page-sequence-runs/<r22 W512 run> --output /home/ikohut/tmp/input-ablations-r23 \
        [--variants d-scan-flags-only e-metadata-only]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
AUXILIARY = ('page_start_head.weight', 'page_start_head.bias', 'page_end_head.weight', 'page_end_head.bias')
# (directory, run-name tag, disabled inputs, page-edge heads)
VARIANTS = [('a-no-edges-metadata', 'noedgeheads-nometadata', ['document_metadata'], False),
            ('b-no-scan-flags', 'noedgeheads-nometadata-noscanflags', ['document_metadata', 'scan_flags'], False),
            ('c-no-page-numbers', 'noedgeheads-nometadata-noscanflags-nopagenumbers',
             ['document_metadata', 'scan_flags', 'page_numbers'], False),
            ('d-scan-flags-only', 'noedgeheads-nometadata-nopagenumbers', ['document_metadata', 'page_numbers'], False),
            ('e-metadata-only', 'boundary5-metadata50-noscanflags-nopagenumbers', ['scan_flags', 'page_numbers'], True),
            ('f-metadata-scan-flags', 'boundary5-metadata50-nopagenumbers', ['page_numbers'], True),
            ('g-metadata-page-numbers', 'boundary5-metadata50-noscanflags', ['scan_flags'], True),
            ('h-metadata-only-noheads', 'noedgeheads-boundary5-metadata50-noscanflags-nopagenumbers',
             ['scan_flags', 'page_numbers'], False),
            ('i-metadata-scan-flags-noheads', 'noedgeheads-boundary5-metadata50-nopagenumbers', ['page_numbers'], False),
            ('j-metadata-page-numbers-noheads', 'noedgeheads-boundary5-metadata50-noscanflags', ['scan_flags'], False),
            ('k-all-signals-noheads', 'noedgeheads-boundary5-metadata50', [], False)]


def digest(state, skip=()):
    h = hashlib.sha256()
    for name, tensor in state.items():
        if name not in skip:
            h.update(name.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def variant(config, tag, disabled, heads, smoke=False):
    config = json.loads(json.dumps(config))
    config['disabled_inputs'] = disabled
    if 'document_metadata' in disabled:
        config['document_metadata_evaluation'] = False
    if not heads:
        config['auxiliary_edges'] = False
        config['loss_weights'] = dict(config['loss_weights'], page_start=0.0, page_end=0.0)
    elif not config.get('auxiliary_edges'):
        raise ValueError('the reference has no page-edge heads')
    if not smoke:
        name = Path(config['run_dir']).name
        if name.count('-boundary5-metadata50-') != 1 or not name.endswith('-r22'):
            raise ValueError('unexpected reference run name')
        name = name.replace('-boundary5-metadata50-', f'-{tag}-').removesuffix('-r22') + '-r23'
        config['run_dir'] = str(Path(config['run_dir']).parent / name)
        config['clearml'] = dict(config['clearml'], name=name)
    return config


def prepare(reference, reference_run, output, names=None):
    selected = [v for v in VARIANTS if names is None or v[0] in names]
    if names is not None and len(selected) != len(set(names)):
        raise ValueError(f'unknown variants: {sorted(set(names) - {v[0] for v in VARIANTS})}')
    if names is None and output.exists():
        raise FileExistsError(output)
    expected = json.loads((reference / 'expected-initialization.json').read_text())['sha256']
    initial = torch.load(reference_run / 'initial.pt', map_location='cpu', weights_only=False)['model']
    if digest(initial) != expected:
        raise ValueError('the reference run does not start from the expected initialization')
    if not set(AUXILIARY) <= set(initial):
        raise ValueError('the reference has no auxiliary heads')
    without_heads = digest(initial, AUXILIARY)
    jobs_file = output / 'jobs.json'
    previous = json.loads(jobs_file.read_text()) if jobs_file.exists() else []
    source = json.loads((reference / 'config.json').read_text())
    smoke = json.loads((reference / 'smoke-config.json').read_text())
    jobs = []
    for directory, tag, disabled, heads in selected:
        root = output / directory
        root.mkdir(parents=True)
        config = variant(source, tag, disabled, heads)
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (root / 'config-diff.json').write_text(json.dumps(
            {k: {'before': source.get(k), 'after': v} for k, v in config.items() if source.get(k) != v}
            | {'derived_from': str(reference)}, indent=2) + '\n')
        test = variant(smoke, tag, disabled, heads, smoke=True)
        test.update(run_dir=str(root / 'smoke-run'), dev_manifest=str(root / 'smoke-dev.jsonl'))
        (root / 'smoke-config.json').write_text(json.dumps(test, indent=2) + '\n')
        (root / 'expected-initialization.json').write_text(json.dumps(
            {'sha256': expected, 'derived_from': f'{reference_run}/initial.pt'} if heads else
            {'sha256': without_heads, 'derived_from': f'{reference_run}/initial.pt without {list(AUXILIARY)}'},
            indent=2) + '\n')
        shutil.copy2(reference / 'smoke-dev.jsonl', root / 'smoke-dev.jsonl')
        shutil.copy2(reference / 'launch.py', root / 'launch.py')
        shutil.copy2(REPOSITORY / 'tools/window_series_smoke.py', root / 'smoke.py')
        jobs.append(dict(id=f'r23-{directory}', root=str(root), run_dir=config['run_dir'], disabled_inputs=disabled,
                         page_edge_heads=heads))
    jobs_file.write_text(json.dumps(previous + jobs, indent=2) + '\n')
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True, help='r22 W512 recipe directory')
    parser.add_argument('--reference-run', type=Path, required=True, help='Its run directory (for initial.pt)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=[v[0] for v in VARIANTS],
                        help='Prepare only these into an existing output directory (default: all, new directory)')
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference, args.reference_run, args.output, args.variants), indent=2))
