"""Freeze W1 S/T/I recipes with the original 512-page sampling policy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


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
    jobs = []
    for tag, slot in [('S', 'nvme'), ('T', 'ssd'), ('I', 'nvme')]:
        root = output / tag
        root.mkdir()
        (root / 'code').symlink_to(output / 'code', target_is_directory=True)
        source = json.loads((reference / tag / 'config.json').read_text())
        config = json.loads(json.dumps(source))
        config.update(window_size=1, sampling_window_size=512, context_microbatch_size=1024)
        name = f'bert-mini-fp16-{slot}-b64parent512-w1-boundary5-metadata50-scattered-mmbert-ocr-lines-stats180-abl{tag}-ftweak005-seed17-r21'
        config['run_dir'] = str(Path(source['run_dir']).parent / name)
        config['inference'] = dict(window_size=1, stride=1, merge='uniform_logits', edge_ramp=1,
                                   edge_ignore=0, edge_power=1, tail='regular',
                                   start_threshold=.5, batch_size=1024)
        config['loss_weights']['starts'] = 0.
        config['final_scan_context_comparison'] = False
        config['clearml']['name'] = name
        write(root / 'config.json', config)
        original_audit = json.loads((reference / tag / 'smoke-modality-audit.json').read_text())
        write(root / 'expected-initialization.json',
              {'sha256': original_audit['initial_parameter_sha256']})
        write(root / 'config-diff.json', {k: {'before': source.get(k), 'after': v}
                                         for k, v in config.items() if source.get(k) != v})
        smoke = json.loads(json.dumps(config))
        smoke.update(run_dir=str(root / 'smoke-run'), pretrain_steps=4, finetune_steps=2,
                     validate_every=2, pretrain_validate_every=2, finetune_validate_every=2,
                     stopping={}, robustness_evaluation=False, pagination_robustness_evaluation=False,
                     final_scan_context_comparison=False)
        smoke['clearml']['enabled'] = False
        smoke['context_evaluation'] = {'enabled': True, 'sample_pages': 32,
                                       'final_sample_pages': 32, 'budget_fraction': .1}
        shutil.copy2(reference / tag / 'smoke-dev.jsonl', root / 'smoke-dev.jsonl')
        smoke['dev_manifest'] = str(root / 'smoke-dev.jsonl')
        write(root / 'smoke-config.json', smoke)
        shutil.copy2(repository / 'tools/context_window_job.py', root / 'launch.py')
        shutil.copy2(repository / 'tools/context_window_smoke.py', root / 'smoke.py')
        jobs.append(dict(id='r21-w1-' + tag, kind='parent', root=str(root), slot=slot,
                         unit='page-sequence-r21-w1-' + tag.lower() + '.service'))
    write(output / 'jobs.json', jobs)
    write(output / 'design.json', dict(reference=str(reference), parent_window_size=512,
          parent_batch_size=64, context_size=1, max_effective_page_slots=32768,
          physical_microbatch=1024, disabled_loss='starts',
          sampling='unchanged parent sampler; augment before splitting; retain all real pages',
          validation='same dev streams, independent pages; pairwise metrics disabled',
          preflight='six real CUDA updates, both phases, audit before each full run'))
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference.resolve(), args.output.resolve(),
                             Path(__file__).resolve().parents[1]), indent=2))
