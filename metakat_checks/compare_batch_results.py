"""Compare CLI evaluation outputs with independent training-repo inference.

Use the same scan order, detected pagination, unknown document metadata and
window merge settings. Native features come from the immutable training store.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from page_sequence.inference import InferenceConfig, tag_pages
from page_sequence.pagination import candidate
from page_sequence.runner import load_checkpoint
from page_sequence.schema import PAGE_TYPES, PLACEMENTS, Page
from training_store.reader import TrainingStore


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model, state = load_checkpoint(args.checkpoint, 'cuda')
    model.eval()
    artifact = torch.load(args.artifact, map_location='cpu', weights_only=True)
    if any(not torch.equal(state['model'][key], value) for key, value in artifact['model'].items()):
        raise ValueError('Training checkpoint and deployed weights differ')
    store = TrainingStore(args.store)
    native = state['provenance']['training_store']
    if store.manifest['build_id'] != native['build_id']:
        raise ValueError('Native store differs from checkpoint provenance')
    cfg = InferenceConfig(window_size=512, stride=448, edge_ramp=48, edge_ignore=12,
                          edge_power=3, batch_size=8, tail='regular', start_threshold=.5)
    types = [('flyleaf' if label == 'FlyLeaf' else label[0].lower()+label[1:]) for label in PAGE_TYPES]
    sides = ['left', 'right', 'singlePage']
    batches = sorted(p for p in args.root.iterdir() if p.is_dir() and (p/'READY.json').is_file())
    if args.limit:
        batches = batches[:args.limit]
    report = {'checkpoint': str(args.checkpoint), 'store': str(args.store),
              'features': 'native training store; MetaKat-detected pagination',
              'context': 'whole scan batch; unknown document boundaries', 'batches': []}
    totals = Counter()
    fields = ['batch', 'page', 'position', 'task', 'metakat', 'native', 'metakat_confidence',
              'native_confidence', 'native_probability_of_metakat_label']
    with (args.output/'disagreements.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for batch in batches:
            output = batch.with_name(batch.name+'.results')
            while not (output/'metakat.json').exists():
                record = output/'cli-run.json'
                if record.exists() and json.loads(record.read_text())['state'] == 'failed':
                    raise RuntimeError('CLI inference failed: '+batch.name)
                if not args.wait:
                    raise RuntimeError('Batch not finished: '+batch.name)
                time.sleep(5)
            manifest = json.loads((batch/'manifest.json').read_text())
            actual = json.loads((output/'metakat.json').read_text())
            by_id = {p['id']: p for p in actual['elements'] if p['type']=='page'}
            pages = []
            row_map = {}
            for p in manifest['pages']:
                current = by_id[p['page']]
                if current['batch_index'] != p['batch_index']:
                    raise ValueError('Scan order differs')
                number = (current.get('pageNumber') or {}).get('text', '')
                page = Page(p['library'], p['document'], p['page'], p['ord'], p['path'], pagenum=number)
                pages.append(page)
                row_map[page.key] = p['row_id']
            def features(current):
                return store.features(store.read_rows([row_map[p.key] for p in current]))
            result = tag_pages(model, features, pages, cfg, 'cuda', pagination_cap=artifact['pagination']['cap'])
            np.savez_compressed(args.output/(batch.name+'.native.npz'), **result)
            stats = {'batch': batch.name, 'pages': len(pages)}
            for task, label_names, probabilities, chosen in (
                ('pageType', types, result['type_probs'], result['type']),
                ('side', sides, result['placement_probs'], result['placement']),
            ):
                errors, deltas = 0, []
                for i, page in enumerate(pages):
                    value = by_id[page.page][task]
                    expected = label_names[int(chosen[i])]
                    actual_label = value['text']
                    deltas.append(abs(float(probabilities[i, label_names.index(actual_label)])-value['confidence']))
                    if actual_label != expected:
                        errors += 1
                        writer.writerow(dict(batch=batch.name, page=page.page, position=i, task=task,
                            metakat=actual_label, native=expected, metakat_confidence=value['confidence'],
                            native_confidence=float(probabilities[i, chosen[i]]),
                            native_probability_of_metakat_label=float(probabilities[i, label_names.index(actual_label)])))
                stats[task] = {'disagreements': errors, 'mean_score_delta': float(np.mean(deltas)),
                               'max_score_delta': float(max(deltas, default=0))}
                totals[task+'_disagreements'] += errors
            errors, deltas = 0, []
            for i, page in enumerate(pages):
                value = by_id[page.page].get('startsDocument')
                if i == 0:
                    if value is not None:
                        raise ValueError('Unexpected first-page boundary prediction')
                    continue
                if value is None:
                    raise ValueError('Missing boundary prediction')
                probability = value['confidence'] if value['val'] else 1.-value['confidence']
                native_probability = float(result['starts_document_probs'][i])
                deltas.append(abs(probability-native_probability))
                if value['val'] != bool(result['starts'][i]):
                    errors += 1
                    writer.writerow(dict(batch=batch.name, page=page.page, position=i, task='startsDocument',
                        metakat=value['val'], native=bool(result['starts'][i]), metakat_confidence=probability,
                        native_confidence=native_probability,
                        native_probability_of_metakat_label=native_probability))
            stats['startsDocument'] = {'disagreements': errors, 'mean_score_delta': float(np.mean(deltas)) if deltas else 0,
                                       'max_score_delta': float(max(deltas, default=0))}
            totals['startsDocument_disagreements'] += errors
            totals['pages'] += len(pages)
            totals['boundaries'] += len(pages)-1
            report['batches'].append(stats)
            report['totals'] = dict(totals)
            atomic_json(args.output/'native-report.json', report)
            stream.flush()
            print(json.dumps(stats), flush=True)
    store.close()


if __name__ == '__main__':
    main()
