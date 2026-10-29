"""Evaluate one checkpoint on the dev streams with and without the optional inputs.

The context model takes two optional inputs: the page numbers (the pagination features) and the document
boundaries (the five-bit metadata with the true document starts and ends of the scan). The four combinations are
run over all dev streams of the manifest, with the checkpoint's own inference settings; the scan endpoints are
inputs in every condition. The report of each condition is the Evaluator's: confusion matrices of the curated
page types, of the DB-agreement of the weak page types and of the DB placements, per-class scores, document
starts and joins, exact spans.

Example:
    python -m page_sequence.evaluate_conditions --checkpoint finetune_best.pt \\
        --manifest /home/ikohut/tmp/page-sequence-data/dev.jsonl --output /home/ikohut/tmp/conditions --device cuda
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .inference import InferenceConfig
from .manifests import file_sha256, load_streams
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--training-store')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--limit-streams', type=int, help='Only the first streams, for testing')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, state = load_checkpoint(args.checkpoint, args.device)
    config, dataset = evaluation_dataset(state, args.training_store)
    inference = InferenceConfig(**config['inference'])
    streams = load_streams(args.manifest, dataset.index, args.split)
    if args.limit_streams:
        streams = streams[:args.limit_streams]
    for name in args.conditions:
        pagination, boundaries = CONDITIONS[name]
        path = args.output / f'{name}.json'
        if path.exists():
            continue
        report = evaluate_streams(model, dataset if pagination else WithoutPagination(dataset), streams, inference,
                                  args.device, provide_document_metadata=boundaries)
        report.update(condition=name, page_numbers=pagination, document_boundaries=boundaries,
                      inference=asdict(inference), checkpoint=str(args.checkpoint),
                      checkpoint_sha256=file_sha256(args.checkpoint), manifest=str(args.manifest),
                      manifest_sha256=file_sha256(args.manifest), checkpoint_step=state['step'])
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(name, f'{report["seconds"]:.0f} s', flush=True)


if __name__ == '__main__':
    main()
