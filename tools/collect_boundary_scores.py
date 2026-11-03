"""Collect selected-checkpoint boundary scores using an experiment's frozen PYTHONPATH.

Only model embeddings and catalogue metadata are read. Each stream's bounded
feature cache is reused between unknown/supplied document-metadata inference.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from page_sequence.data import page_edge_targets
from page_sequence.inference import InferenceConfig, predict_stream
from page_sequence.manifests import file_sha256, load_streams
from page_sequence.runner import evaluation_dataset, load_checkpoint


class StreamCache:
    def __init__(self, dataset, limit=64 << 20):
        self.dataset, self.limit, self.used, self.cache = dataset, limit, 0, {}

    def __getitem__(self, window):
        key = window
        if key not in self.cache:
            value = self.dataset[window]
            size = sum(v.nbytes for v in value['features'].values())
            if self.used + size <= self.limit:
                self.cache[key] = value
                self.used += size
        else:
            value = self.cache[key]
        # Supplied inference replaces the features mapping, never these arrays.
        return dict(value, features=dict(value['features']))


def run(args):
    torch.set_num_threads(1)
    runtime = Path(args.runtime)
    config = json.loads((runtime / 'config.json').read_text())
    checkpoint = Path(config['run_dir']) / 'finetune_best.pt'
    model, state = load_checkpoint(checkpoint, 'cuda')
    config, dataset = evaluation_dataset(state, args.training_store)
    streams = load_streams(config['dev_manifest'], dataset.index, 'dev')
    inference = InferenceConfig(**config['inference'])
    selected_step = state['step']
    expected = next(json.loads(line)['dev'] for line in
                    (Path(config['run_dir']) / 'validation.jsonl').open()
                    if json.loads(line)['step'] == selected_step)
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                    checkpoint_step=selected_step, manifest_sha256=file_sha256(config['dev_manifest']),
                    index_sha256=state['provenance']['index_sha256'], split='dev',
                    frozen_inference=__import__('page_sequence.inference', fromlist=['']).__file__,
                    inference=asdict(inference), streams=len(streams),
                    expected_unknown=expected['starts']['all'],
                    expected_supplied=expected['metadata_provided']['starts']['all'])
    del state
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recorded = output / 'metadata.json'
    if recorded.exists() and json.loads(recorded.read_text()) != metadata:
        raise ValueError('existing score collection belongs to a different evaluation')
    recorded.write_text(json.dumps(metadata, indent=2) + '\n')
    start = time.monotonic()
    for i, stream in enumerate(streams):
        path = output / f'{i:06d}.npz'
        if path.exists():
            continue
        rows = stream['rows']
        targets = page_edge_targets(dataset.index, rows)
        assert np.isin(targets['page_start'], [0, 1]).all()
        cache = StreamCache(dataset)
        scores = {}
        for mode in ('unknown', 'supplied'):
            supplied = (dict(document_starts=targets['page_start'], document_ends=targets['page_end'])
                        if mode == 'supplied' else {})
            result = predict_stream(model, cache, rows, inference, 'cuda', **supplied)
            mask = np.asarray(result['pair_mask'], bool)
            assert not mask[0] and mask[1:].all()
            scores[mode] = result['starts_document_probs'][mask].copy()
            assert np.isfinite(scores[mode]).all()
            del result
        with path.with_suffix('.partial').open('wb') as f:
            np.savez_compressed(f, rows=np.asarray(rows[1:], np.int64),
                                previous_rows=np.asarray(rows[:-1], np.int64),
                                truth=targets['page_start'][1:].astype(bool), **scores)
        path.with_suffix('.partial').replace(path)
        del cache, scores
        if (i + 1) % 10 == 0 or i + 1 == len(streams):
            print(json.dumps(dict(streams_done=i + 1, streams=len(streams),
                                  seconds=round(time.monotonic() - start, 2))), flush=True)
    for mode in ('unknown', 'supplied'):
        tp = fp = fn = total = 0
        for path in sorted(output.glob('*.npz')):
            with np.load(path) as values:
                truth, predicted = values['truth'], values[mode] >= .5
                tp += int((truth & predicted).sum())
                fp += int((~truth & predicted).sum())
                fn += int((truth & ~predicted).sum())
                total += len(truth)
        expected = metadata['expected_' + mode]
        actual = dict(n=total, tp=tp, fp=fp, fn=fn)
        if any(actual[k] != expected[k] for k in actual):
            raise ValueError(f'{mode} does not reproduce selected-checkpoint validation: {actual} != {expected}')
    (output / 'complete.json').write_text(json.dumps(dict(verified_default_counts=True,
                                                         seconds=time.monotonic() - start)) + '\n')
    print(json.dumps(dict(complete=True, output=str(output))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-store', required=True)
    run(parser.parse_args())
