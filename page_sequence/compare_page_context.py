"""Paired page-type evaluation: isolated documents versus one continuous stream."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluation import classification_metrics, front_back_metrics
from .inference import InferenceConfig, predict_stream, window_spans
from .manifests import file_sha256, load_streams
from .runner import evaluation_dataset, load_checkpoint
from .schema import IGNORE, PAGE_TYPES


def page_panel(index, streams):
    """Preserve manifest order; require unique pages and complete document runs."""
    rows = tuple(row for stream in streams for row in stream['rows'])
    if not rows or len(set(rows)) != len(rows):
        raise ValueError('nonempty, nonrepeated page panel required')
    labels = np.full(len(rows), IGNORE, np.int16)
    gold = np.zeros(len(rows), bool)
    spans, previous, start = [], None, 0
    documents = {}
    for lo in range(0, len(rows), 4096):
        chunk = rows[lo:lo + 4096]
        documents.update(((d.library, d.doc), d) for d in index.documents_for_rows(chunk))
        for i, page in enumerate(index.pages(chunk), lo):
            labels[i], gold[i], _ = page.target()
            if previous is not None and page.doc_key != previous:
                spans.append((start, i))
                start = i
            previous = page.doc_key
    spans.append((start, len(rows)))
    by_first = {d.first: d for d in documents.values()}
    seen = set()
    for lo, hi in spans:
        d = by_first.get(rows[lo])
        if (d is None or d.id in seen or hi - lo != d.length
                or any(row != d.first + i for i, row in enumerate(rows[lo:hi]))):
            raise ValueError('panel must contain complete, contiguous documents exactly once')
        seen.add(d.id)
    return rows, spans, labels, gold


def type_scores(labels, prediction, mask):
    n = len(PAGE_TYPES)
    cm = np.bincount(labels[mask].astype(np.int64) * n + prediction[mask],
                     minlength=n * n).reshape(n, n)
    return dict(classification_metrics(cm, PAGE_TYPES), front_back_confusion=front_back_metrics(cm))


def paired_scores(labels, gold, isolated, continuous):
    result = {'all_pages_changed': int(np.sum(isolated != continuous)), 'pages': len(labels)}
    for name, mask in (('curated_type', gold & (labels != IGNORE)),
                       ('weak_type_agreement', ~gold & (labels != IGNORE))):
        a, b, y = isolated[mask], continuous[mask], labels[mask]
        result[name] = {
            'n': len(y), 'changed': int(np.sum(a != b)),
            'continuous_fixes': int(np.sum((a != y) & (b == y))),
            'continuous_regressions': int(np.sum((a == y) & (b != y))),
            'both_correct': int(np.sum((a == y) & (b == y))),
            'both_wrong': int(np.sum((a != y) & (b != y))),
        }
    return result


def compare(model, dataset, streams, inference, output, device='cpu', provide_document_metadata=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, spans, labels, gold = page_panel(dataset.index, streams)
    np.savez(output / 'panel.npz', rows=np.asarray(rows, np.int64),
             document_spans=np.asarray(spans, np.int64), labels=labels, gold=gold,
             page_types=np.asarray(PAGE_TYPES))
    report = {'pages': len(rows), 'documents': len(spans), 'inference': asdict(inference),
              'continuous_order': 'source manifest streams concatenated in their recorded order',
              'modes': {}}
    report['document_metadata_provided'] = provide_document_metadata
    starts = np.zeros(len(rows), np.float32)
    ends = np.zeros(len(rows), np.float32)
    for lo, hi in spans:
        starts[lo], ends[hi - 1] = 1, 1
    predictions = {}
    for mode, runs in (('per_document', spans), ('continuous', [(0, len(rows))])):
        start = time.perf_counter()
        probabilities = np.lib.format.open_memmap(output / f'{mode}-probabilities.npy', mode='w+',
                                                  dtype=np.float32, shape=(len(rows), len(PAGE_TYPES)))
        processed = windows = full = 0
        for j, (lo, hi) in enumerate(runs):
            metadata = (dict(document_starts=starts[lo:hi], document_ends=ends[lo:hi])
                        if provide_document_metadata else {})
            result = predict_stream(model, dataset, rows[lo:hi], inference, device, **metadata)
            probabilities[lo:hi] = result['type_probs']
            processed += result['processed_pages']
            border_tokens = getattr(model, 'scan_borders', False)
            if border_tokens or getattr(model, 'boundary_features', False):
                from .scan_borders import scan_windows
                tiles = scan_windows(hi - lo, inference.window_size, inference.stride, border_tokens)
            else:
                tiles = [(s, e, False, False) for s, e in window_spans(hi - lo, inference.window_size, inference.stride)]
            windows += len(tiles)
            full += sum(e - s + (begin + end) * border_tokens == inference.window_size
                        for s, e, begin, end in tiles)
            del result
            if j % 100 == 0 or j + 1 == len(runs):
                print(json.dumps({'mode': mode, 'runs': j + 1, 'total_runs': len(runs),
                                  'pages': hi, 'seconds': time.perf_counter() - start}), flush=True)
        probabilities.flush()
        prediction = np.asarray(probabilities.argmax(axis=1))
        del probabilities
        predictions[mode] = prediction
        scores = {name: type_scores(labels, prediction, mask) for name, mask in (
            ('curated_type', gold & (labels != IGNORE)),
            ('weak_type_agreement', ~gold & (labels != IGNORE)))}
        short = np.zeros(len(rows), bool)
        edges = np.zeros(len(rows), bool)
        for lo, hi in spans:
            short[lo:hi] = hi - lo <= inference.window_size
            edges[lo:min(hi, lo + 16)] = True
            edges[max(lo, hi - 16):hi] = True
        scores['curated_strata'] = {name: type_scores(labels, prediction, mask & gold & (labels != IGNORE))
                                    for name, mask in (('document_fits_window', short),
                                                       ('document_exceeds_window', ~short),
                                                       ('first_or_last_16', edges), ('interior', ~edges))}
        scores.update(seconds=time.perf_counter() - start, processed_pages=processed,
                      windows=windows, full_windows=full)
        report['modes'][mode] = scores
        (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    report['paired'] = paired_scores(labels, gold, predictions['per_document'], predictions['continuous'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='New directory for report and aligned probabilities')
    parser.add_argument('--training-store')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, state = load_checkpoint(args.checkpoint, args.device)
    config, dataset = evaluation_dataset(state, args.training_store)
    streams = load_streams(args.manifest, dataset.index, 'dev')
    settings = dict(config['inference'], batch_size=args.batch_size)
    inference = InferenceConfig(**settings)
    # Optimizer/RNG/sampler state is unnecessary for read-only inference.
    metadata = {'checkpoint': str(Path(args.checkpoint).resolve()),
                'checkpoint_sha256': file_sha256(args.checkpoint), 'checkpoint_step': state['step'],
                'checkpoint_best': state['best'], 'manifest_sha256': file_sha256(args.manifest),
                'index_sha256': state['provenance']['index_sha256'],
                'comparison_code_sha256': file_sha256(__file__), 'split': 'dev'}
    del state
    report = dict(metadata, **compare(model, dataset, streams, inference, args.output, args.device))
    (Path(args.output) / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': args.output, 'paired': report['paired']}), flush=True)


if __name__ == '__main__':
    main()
