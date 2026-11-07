"""Evaluate one checkpoint on the dev streams with and without the optional inputs.

The context model takes two optional inputs: the page numbers (the pagination features) and the document
boundaries (the five-bit metadata with the true document starts and ends of the scan). The four combinations are
run over all dev streams of the manifest, with the checkpoint's own inference settings; the scan endpoints are
inputs in every condition. The report of each condition is the Evaluator's: confusion matrices of the curated
page types, of the DB-agreement of the weak page types and of the DB placements, per-class scores, document
starts and joins, exact spans.

`--modalities` additionally evaluates the checkpoint with only some of its main signals (image/text/stats):
the others are masked as in a modality ablation, before inference (vectors and presence bits zero). Each subset
must be part of the checkpoint's own enabled modalities; `all` expands to every nonempty subset of them. The
reports are then written to `<output>/<subset>/<condition>.json`, the subset named like `image+text`.

`--inference deployment` replaces the checkpoint's validation merging by the policy deployed in MetaKat, scaled to
the checkpoint's window (`inference.deployment_inference`): overlap W/8, regular tail, cubic centre weights with a
ramp of 3/4 and an ignored edge of 3/16 of the overlap.

Example:
    python -m page_sequence.evaluate_conditions --checkpoint finetune_best.pt \\
        --manifest /home/ikohut/tmp/page-sequence-data/dev.jsonl --output /home/ikohut/tmp/conditions --device cuda
"""
import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .inference import InferenceConfig, deployment_inference
from .manifests import file_sha256, load_streams
from .modalities import ModalitySubsetDataset, enabled_modalities
from .runner import evaluate_streams, evaluation_dataset, load_checkpoint

CONDITIONS = {'none': (False, False), 'page_numbers': (True, False), 'boundaries': (False, True),
              'both': (True, True)}


class WithoutPagination:
    """The dataset with the pagination features zeroed, i.e. the page numbers missing."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, window):
        example = self._dataset[window]
        if 'pagination' not in example['features']:
            raise ValueError('the checkpoint has no pagination features to omit')
        features = dict(example['features'], pagination=np.zeros_like(example['features']['pagination']))
        return dict(example, features=features)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def modality_subsets(requested, enabled):
    """The requested subsets of the checkpoint's enabled modalities, in canonical order."""
    if requested == ['all']:
        return [subset for k in range(len(enabled), 0, -1) for subset in combinations(enabled, k)]
    subsets = [enabled_modalities(name.split('+')) for name in requested]
    for subset in subsets:
        if set(subset) - set(enabled):
            raise ValueError(f'checkpoint handles only {"+".join(enabled)}, not {"+".join(subset)}')
    return subsets


INFERENCE_POLICIES = ('checkpoint', 'deployment')


def evaluate_checkpoint(checkpoint, manifest, output, conditions=tuple(CONDITIONS), modalities=None, split='dev',
                        training_store=None, device='cpu', limit_streams=None, inference='checkpoint'):
    """Write one report per (modality subset, condition) that is not there yet; return the written paths."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, state = load_checkpoint(checkpoint, device)
    config, dataset = evaluation_dataset(state, training_store)
    saved = InferenceConfig(**config['inference'])
    if inference not in INFERENCE_POLICIES:
        raise ValueError(f'inference must be one of {INFERENCE_POLICIES}')
    policy = inference
    inference = saved if policy == 'checkpoint' else deployment_inference(
        saved.window_size, saved.batch_size, saved.start_threshold)
    streams = load_streams(manifest, dataset.index, split)
    if limit_streams:
        streams = streams[:limit_streams]
    enabled = enabled_modalities(config.get('enabled_modalities'))
    subsets = modality_subsets(list(modalities), enabled) if modalities else [None]
    written = []
    for subset, name in [(subset, name) for subset in subsets for name in conditions]:
        pagination, boundaries = CONDITIONS[name]
        directory = output if subset is None else output / '+'.join(subset)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{name}.json'
        if path.exists():
            continue
        data = dataset if subset is None else ModalitySubsetDataset(dataset, subset)
        report = evaluate_streams(model, data if pagination else WithoutPagination(data), streams, inference,
                                  device, provide_document_metadata=boundaries)
        report.update(condition=name, page_numbers=pagination, document_boundaries=boundaries,
                      checkpoint_modalities=list(enabled),
                      modalities=list(enabled if subset is None else subset),
                      inference=asdict(inference), inference_policy=policy, checkpoint=str(checkpoint),
                      checkpoint_sha256=file_sha256(checkpoint), manifest=str(manifest),
                      manifest_sha256=file_sha256(manifest), checkpoint_step=state['step'],
                      streams_evaluated=len(streams))
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        written.append(path)
        print(name if subset is None else '+'.join(subset) + ' ' + name, f'{report["seconds"]:.0f} s', flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--training-store')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--modalities', nargs='+', metavar='SUBSET',
                        help='Main-signal subsets such as image+text, or all; default: the checkpoint\'s own set')
    parser.add_argument('--inference', choices=INFERENCE_POLICIES, default='checkpoint',
                        help='Merging policy: the checkpoint\'s validation settings or the deployed MetaKat policy')
    parser.add_argument('--limit-streams', type=int, help='Only the first streams, for testing')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.manifest, args.output, args.conditions, args.modalities,
                        args.split, args.training_store, args.device, args.limit_streams, args.inference)


if __name__ == '__main__':
    main()
