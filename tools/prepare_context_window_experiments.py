"""Freeze W1 recipes with the original 512-page sampling policy: S/T/I, then (`--add-combinations`) ST/SI/TI/STI."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


STORES = {'nvme': '/home/ikohut/tmp/document-store-ocr-lines-sparse-v2',
          'ssd': '/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/document-store-ocr-lines-sparse-v2'}
LETTERS = {'image': 'I', 'text': 'T', 'stats': 'S'}
# Later W1 combinations: (tag, slot, r20 recipe, enabled modalities, dependency). STI has no r20
# recipe (its reference is r19, the same config without enabled_modalities), so it is SI's recipe
# with all three signals enabled explicitly. Slot chains: NVMe T -> ST -> TI, SSD I -> SI -> STI.
COMBINATIONS = [('ST', 'nvme', 'ST', ['text', 'stats'], 'r21-w1-T'),
                ('SI', 'ssd', 'SI', ['image', 'stats'], 'r21-w1-I'),
                ('TI', 'nvme', 'TI', ['image', 'text'], 'r21-w1-ST'),
                ('STI', 'ssd', 'SI', ['image', 'text', 'stats'], 'r21-w1-SI')]


def prepare_job(reference, output, repository, tag, slot, source_tag, depends_on, modalities=None):
    root = output / tag
    root.mkdir()
    (root / 'code').symlink_to(output / 'code', target_is_directory=True)
    source = json.loads((reference / source_tag / 'config.json').read_text())
    config = json.loads(json.dumps(source))
    if modalities is not None:
        config['enabled_modalities'] = modalities
    if ''.join(sorted(LETTERS[m] for m in config['enabled_modalities'])) != ''.join(sorted(tag)):
        raise ValueError(f'{tag} does not match its enabled modalities')
    if modalities is not None:
        # Relocate to the native store on the job's disk; both copies are identical.
        config['index'] = STORES[slot] + '/catalogue.sqlite'
        config['training_store'] = dict(config['training_store'], path=STORES[slot])
    config.update(window_size=1, sampling_window_size=512, context_microbatch_size=32768)
    name = f'bert-mini-fp16-{slot}-b64parent512-w1-boundary5-metadata50-scattered-mmbert-ocr-lines-stats180-abl{tag}-ftweak005-seed17-r21'
    config['run_dir'] = str(Path(source['run_dir']).parent / name)
    config['inference'] = dict(window_size=1, stride=1, merge='uniform_logits', edge_ramp=1,
                               edge_ignore=0, edge_power=1, tail='regular',
                               start_threshold=.5, batch_size=1024)
    config['loss_weights']['starts'] = 0.
    config['final_scan_context_comparison'] = False
    config['clearml']['name'] = name
    write(root / 'config.json', config)
    original_audit = json.loads((reference / source_tag / 'smoke-modality-audit.json').read_text())
    write(root / 'expected-initialization.json',
          {'sha256': original_audit['initial_parameter_sha256']})
    write(root / 'config-diff.json', {k: {'before': source.get(k), 'after': v}
                                     for k, v in config.items() if source.get(k) != v}
          | ({'derived_from': str(reference / source_tag)} if source_tag != tag else {}))
    smoke = json.loads(json.dumps(config))
    smoke.update(run_dir=str(root / 'smoke-run'), pretrain_steps=4, finetune_steps=2,
                 validate_every=2, pretrain_validate_every=2, finetune_validate_every=2,
                 stopping={}, robustness_evaluation=False, pagination_robustness_evaluation=False,
                 final_scan_context_comparison=False)
    smoke['clearml']['enabled'] = False
    smoke['context_evaluation'] = {'enabled': True, 'sample_pages': 32,
                                   'final_sample_pages': 32, 'budget_fraction': .1}
    shutil.copy2(reference / source_tag / 'smoke-dev.jsonl', root / 'smoke-dev.jsonl')
    smoke['dev_manifest'] = str(root / 'smoke-dev.jsonl')
    write(root / 'smoke-config.json', smoke)
    shutil.copy2(repository / 'tools/context_window_job.py', root / 'launch.py')
    shutil.copy2(repository / 'tools/context_window_smoke.py', root / 'smoke.py')
    return dict(id='r21-w1-' + tag, kind='parent', root=str(root), slot=slot,
                unit='page-sequence-r21-w1-' + tag.lower() + '.service', depends_on=[depends_on])


def add_combinations(reference, output, repository):
    """Add the mixed-signal W1 recipes to an existing r21 directory, sharing its frozen code."""
    jobs = [prepare_job(reference, output, repository, tag, slot, source_tag, depends_on, modalities)
            for tag, slot, source_tag, modalities, depends_on in COMBINATIONS]
    write(output / 'jobs-combinations.json', jobs)
    return jobs


def prepare(reference, output, repository):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Running r20 code is copied, never modified. Only audited W1 support is overlaid.
    shutil.copytree(reference / 'code', output / 'code',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = ['context_training.py', 'context_evaluation.py', 'evaluation.py',
             'inference.py', 'losses.py', 'runner.py']
    for name in files:
        shutil.copy2(repository / 'page_sequence' / name, output / 'code/page_sequence' / name)
    checksums = {str(p.relative_to(output / 'code')): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (output / 'code').rglob('*') if p.is_file()}
    write(output / 'code-checksums.json', checksums)
    jobs = [prepare_job(reference, output, repository, tag, slot, tag,
                        {'S': 'r20-SI', 'T': 'r20-TI', 'I': 'r21-w1-S'}[tag])
            for tag, slot in [('S', 'nvme'), ('T', 'ssd'), ('I', 'nvme')]]
    write(output / 'jobs.json', jobs)
    write(output / 'design.json', dict(reference=str(reference), parent_window_size=512,
          parent_batch_size=64, context_size=1, max_effective_page_slots=32768,
          physical_microbatch=32768, disabled_loss='starts',
          sampling='unchanged parent sampler; augment before splitting; retain all real pages',
          validation='same dev streams, independent pages; pairwise metrics disabled',
          preflight='six real CUDA updates, both phases, audit before each full run'))
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--add-combinations', action='store_true',
                        help='Add ST/SI/TI/STI to an existing output directory')
    args = parser.parse_args()
    function = add_combinations if args.add_combinations else prepare
    print(json.dumps(function(args.reference.resolve(), args.output.resolve(),
                              Path(__file__).resolve().parents[1]), indent=2))
