"""Freeze the r22 context-window series: full-modality STI at W512, 256, 128, 64, 32, 16 and 8.

Every run starts from the r19 STI W512 recipe and keeps its 512-page sampler, 64 sampled windows per update,
losses and stages. Every W trains on the same sampled data; only the context differs:
- the sampler tiles complete (coverage) scans into 512-page windows without overlap (stride 512, regular tail);
  anchor windows are sampled as before;
- every sampled window, coverage or anchor, is cut into consecutive pieces of at most W, never between a
  document start and its previous page (`context_boundary_aware_cuts`); at W512 the windows stay whole. All
  pieces of an update are trained together in physical batches of at most 32,768 positions with one optimizer
  step;
- fixed phase lengths without early stopping (15,427 pretraining and 4,000 fine-tuning updates), and input
  dropout seeded per update (`augmentation_seed_per_step`), so the dropout is the same for every W;
- validation and final evaluation merge windows with the deployed MetaKat policy scaled to W
  (`inference.deployment_inference`: overlap W/8, cubic ramp 3/4 and ignored edge 3/16 of the overlap).

The source is frozen from a committed revision (`git archive`), the store copy follows the job's disk slot.
Registration in the queue monitor is a separate step.

Example:
    python tools/prepare_window_series.py --reference /home/ikohut/tmp/stats-fine-groups-r19/w512 \\
        --output /home/ikohut/tmp/window-series-r22
"""
import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from page_sequence.inference import deployment_inference  # noqa: E402

STORES = {'nvme': '/home/ikohut/tmp/document-store-ocr-lines-sparse-v2',
          'ssd': '/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/document-store-ocr-lines-sparse-v2'}
PARENT, PHYSICAL = 512, 32768
# Pretraining keeps the r19 budget as a fixed length (most runs selected 14,000-15,427). Fine-tuning runs stopped
# at about 2,250 updates with their best at 250-3,500; it gets a fixed, slightly longer 4,000.
PRETRAIN_STEPS, FINETUNE_STEPS = 15427, 4000
# (window, slot, dependency): larger windows first, the two disks alternating after the running W1 jobs.
JOBS = [(512, 'nvme', 'r21-w1-STI'), (256, 'ssd', 'r21-w1-SI'), (128, 'nvme', 'r22-w512'),
        (64, 'ssd', 'r22-w256'), (32, 'nvme', 'r22-w128'), (16, 'ssd', 'r22-w64'), (8, 'nvme', 'r22-w32')]
INITIAL_PARAMETERS = '0be254f8fe96c2fb406c28b99150f9a9ea5a67a46ba4b85d7733e84769aad527'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def freeze_code(repository, output):
    if subprocess.run(['git', 'status', '--porcelain', '--', 'page_sequence'], cwd=repository,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise RuntimeError('page_sequence has uncommitted changes; commit before freezing')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository, capture_output=True, text=True,
                            check=True).stdout.strip()
    archive = subprocess.run(['git', 'archive', commit, 'page_sequence'], cwd=repository,
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(output / 'code', filter='data')
    return commit


def recipe(reference, window, slot):
    config = json.loads((reference / 'config.json').read_text())
    config['enabled_modalities'] = ['image', 'text', 'stats']
    config['index'] = STORES[slot] + '/catalogue.sqlite'
    config['training_store'] = dict(config['training_store'], path=STORES[slot])
    # Complete scans are tiled without overlap (stride 512, regular tail): every scan page once per scan.
    config['scan_batches'] = dict(config['scan_batches'], stride=PARENT, tail='regular')
    config['inference'] = asdict(deployment_inference(window, batch_size=PHYSICAL // window,
                                                      start_threshold=config['inference']['start_threshold']))
    # The same path for every W, 512 included: every sampled window, coverage or anchor, is cut into
    # consecutive pieces of at most W, never separating a document start from its previous page.
    config.update(window_size=window, sampling_window_size=PARENT, context_microbatch_size=PHYSICAL,
                  context_boundary_aware_cuts=True, augmentation_seed_per_step=True)
    # Fixed phase lengths, no early stopping: every W trains on the same data. The best checkpoint of each
    # phase is still selected by validation (pretraining only after the first coverage epoch, as in r19).
    if config['stopping'] != {'pretrain': {'min_coverage_epochs': 1, 'patience': 5}, 'finetune': {'patience': 5}}:
        raise ValueError('unexpected reference stopping policy')
    config.update(pretrain_steps=PRETRAIN_STEPS, finetune_steps=FINETUNE_STEPS,
                  stopping={'pretrain': {'min_coverage_epochs': 1}})
    name = (f'bert-mini-fp16-{slot}-b64parent512-w{window}-boundary5-metadata50-scattered-mmbert-ocr-lines-stats180-'
            f'tailregular-ftweak005-seed17-r22')
    config['run_dir'] = str(Path(config['run_dir']).parent / name)
    config['clearml'] = dict(config['clearml'], name=name)
    return config


def prepare_job(reference, output, repository, window, slot, depends_on):
    root = output / f'w{window}'
    root.mkdir()
    (root / 'code').symlink_to(output / 'code', target_is_directory=True)
    source = json.loads((reference / 'config.json').read_text())
    config = recipe(reference, window, slot)
    write(root / 'config.json', config)
    write(root / 'config-diff.json', {k: {'before': source.get(k), 'after': v}
                                     for k, v in config.items() if source.get(k) != v}
          | {'derived_from': str(reference)})
    write(root / 'expected-initialization.json', {'sha256': INITIAL_PARAMETERS})
    smoke = json.loads(json.dumps(config))
    smoke.update(run_dir=str(root / 'smoke-run'), pretrain_steps=4, finetune_steps=2,
                 validate_every=2, pretrain_validate_every=2, finetune_validate_every=2, stopping={})
    smoke['clearml']['enabled'] = False
    smoke['context_evaluation'] = {'enabled': True, 'sample_pages': 32,
                                   'final_sample_pages': 32, 'budget_fraction': .1}
    shutil.copy2(reference / 'smoke-dev.jsonl', root / 'smoke-dev.jsonl')
    smoke['dev_manifest'] = str(root / 'smoke-dev.jsonl')
    write(root / 'smoke-config.json', smoke)
    shutil.copy2(repository / 'tools/context_window_job.py', root / 'launch.py')
    shutil.copy2(repository / 'tools/window_series_smoke.py', root / 'smoke.py')
    return dict(id=f'r22-w{window}', kind='parent', root=str(root), slot=slot,
                unit=f'page-sequence-r22-w{window}.service', depends_on=[depends_on])


def prepare(reference, output, repository):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    commit = freeze_code(repository, output)
    write(output / 'source-commit.json', {'commit': commit})
    jobs = [prepare_job(reference, output, repository, *job) for job in JOBS]
    write(output / 'jobs.json', jobs)
    write(output / 'design.json', dict(
        reference=str(reference), parent_window_size=PARENT, parent_batch_size=64,
        windows=[w for w, _, _ in JOBS], max_physical_positions=PHYSICAL,
        sampler='r19 sampler; complete scans tiled into 512-page windows without overlap, regular tail',
        cutting='every sampled window cut into consecutive pieces of at most W, never separating a document '
                'start from its previous page; each page once',
        phases=dict(pretrain_steps=PRETRAIN_STEPS, finetune_steps=FINETUNE_STEPS, early_stopping=False),
        augmentation_randomness='seeded per update, independent of the model and W',
        augmentation='drawn per sampled window before cutting',
        evaluation='deployed MetaKat merging scaled to W: overlap W/8, cubic ramp 3/4, ignored edge 3/16',
        preflight='six real CUDA updates, both phases and final evaluations, audited before each full run'))
    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference.resolve(), args.output.resolve(),
                             Path(__file__).resolve().parents[1]), indent=2))
