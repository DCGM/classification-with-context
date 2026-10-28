"""Compare overlap sizes on identical pages, continuously or per document."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compare_page_context import page_panel, paired_scores, type_scores
from .evaluation import binary_metrics, classification_metrics
from .inference import InferenceConfig, predict_stream, window_spans
from .manifests import file_sha256, load_streams
from .runner import evaluation_dataset, load_checkpoint
from .schema import IGNORE, PAGE_TYPES, PLACEMENTS


def sweep(model, dataset, streams, inference, overlaps, output, device='cpu',
          processing='continuous'):
    if processing not in ('continuous', 'per_document'):
        raise ValueError('unknown processing mode')
    overlaps = list(overlaps)
    if not overlaps or len(set(overlaps)) != len(overlaps) or any(
            not 1 <= n < inference.window_size for n in overlaps):
        raise ValueError('distinct overlaps in 1..window_size-1 required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, spans, labels, gold = page_panel(dataset.index, streams)
    np.savez(output / 'panel.npz', rows=np.asarray(rows, np.int64),
             document_spans=np.asarray(spans, np.int64), labels=labels, gold=gold,
             page_types=np.asarray(PAGE_TYPES))
    starts, ends = np.zeros(len(rows), np.float32), np.zeros(len(rows), np.float32)
    for lo, hi in spans:
        starts[lo], ends[hi - 1] = 1, 1
    sides = np.full(len(rows), IGNORE, np.int16)
    for lo in range(0, len(rows), 4096):
        sides[lo:lo + 4096] = [p.target()[2] for p in dataset.index.pages(rows[lo:lo + 4096])]
    side_mask = sides != IGNORE
    runs = spans if processing == 'per_document' else [(0, len(rows))]
    long_document = np.zeros(len(rows), bool)
    for lo, hi in spans:
        long_document[lo:hi] = hi - lo > inference.window_size
    report = dict(pages=len(rows), documents=len(spans), baseline_overlap=overlaps[0],
                  processing=processing,
                  continuous_order='source manifest streams concatenated in recorded order',
                  results={})
    for provided in (False, True):
        mode = 'provided' if provided else 'unknown'
        report['results'][mode] = {}
        baseline = None
        for overlap in overlaps:
            settings = replace(inference, stride=inference.window_size - overlap)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = dict(type_probs=np.empty((len(rows), len(PAGE_TYPES)), np.float32),
                          placement=np.empty(len(rows), np.int64),
                          starts_document_probs=np.empty(len(rows), np.float32),
                          pair_mask=np.empty(len(rows), bool), processed_pages=0)
            for lo, hi in runs:
                metadata = (dict(document_starts=starts[lo:hi], document_ends=ends[lo:hi])
                            if provided else {})
                current = predict_stream(model, dataset, rows[lo:hi], settings, device, **metadata)
                for name in ('type_probs', 'placement', 'starts_document_probs', 'pair_mask'):
                    result[name][lo:hi] = current[name]
                result['processed_pages'] += current['processed_pages']
                del current
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            prediction = result['type_probs'].argmax(-1)
            if baseline is None:
                baseline = prediction.copy()
            mask = result['pair_mask']
            cm = np.bincount(sides[side_mask].astype(np.int64) * 3 + result['placement'][side_mask],
                             minlength=9).reshape(3, 3)
            entry = dict(inference=asdict(settings), seconds=seconds,
                         processed_pages=result['processed_pages'], pages_per_second=len(rows) / seconds,
                         windows=sum(len(window_spans(hi - lo, settings.window_size,
                                                      settings.stride, settings.tail))
                                     for lo, hi in runs),
                         curated_type=type_scores(labels, prediction, gold & (labels != IGNORE)),
                         curated_long_documents=type_scores(
                             labels, prediction, gold & (labels != IGNORE) & long_document),
                         weak_type_agreement=type_scores(labels, prediction, ~gold & (labels != IGNORE)),
                         placement_db_agreement=classification_metrics(cm, PLACEMENTS),
                         starts=binary_metrics(starts[mask], result['starts_document_probs'][mask],
                                               settings.start_threshold),
                         paired_against_baseline=paired_scores(labels, gold, baseline, prediction))
            np.save(output / f'{mode}-overlap{overlap}-type-probabilities.npy', result['type_probs'])
            report['results'][mode][str(overlap)] = entry
            (output / 'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps(dict(mode=mode, overlap=overlap, seconds=seconds,
                                  type_f1=entry['curated_type']['weighted_f1'],
                                  boundary_f1=entry['starts']['f1'])), flush=True)
            del result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-store')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--overlaps', nargs='+', type=int, default=[256, 128, 64, 32, 16])
    parser.add_argument('--processing', choices=['continuous', 'per_document'], default='continuous')
    parser.add_argument('--tail', choices=['anchored', 'regular'],
                        help='Override final-window placement; regular keeps stride and permits a short tail')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, state = load_checkpoint(args.checkpoint, args.device)
    config, dataset = evaluation_dataset(state, args.training_store)
    streams = load_streams(args.manifest, dataset.index, 'dev')
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_step=state['step'],
                    manifest_sha256=file_sha256(args.manifest),
                    index_sha256=state['provenance']['index_sha256'],
                    comparison_code_sha256=file_sha256(__file__))
    del state
    settings = InferenceConfig(**dict(config['inference'], batch_size=args.batch_size))
    if args.tail is not None:
        settings = replace(settings, tail=args.tail)
    report = dict(metadata, **sweep(model, dataset, streams, settings, args.overlaps, args.output,
                                   args.device, args.processing))
    (Path(args.output) / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
