"""Final-checkpoint diagnostics on bounded dev panels; never used for selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .context_evaluation import context_blocks, evaluate_contexts


def corrupt_examples(examples, modality_probability=.1, page_probability=.05, seed=719, apply=True):
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in (modality_probability, page_probability)):
        raise ValueError('corruption probabilities must be finite and in [0, 1]')
    rows = np.unique([row for example in examples for row in example['rows']])
    # Unique catalogue identities share masks, including repeated panel pages.
    draws = np.random.default_rng(seed).random((len(rows), 4))
    page_missing = draws[:, 0] < page_probability
    modality_keep = draws[:, 1:] >= modality_probability
    for example in examples:
        positions = np.searchsorted(rows, example['rows'])
        missing = page_missing[positions]
        example['diagnostic_mask'] = missing
        if not apply:
            continue
        features = dict(example['features'])
        keep = modality_keep[positions] & ~missing[:, None]
        features['presence'] = features['presence'] * keep
        for i, name in enumerate(('image', 'text', 'stats')):
            features[name] = features[name] * keep[:, i:i + 1]
        if 'pagination' in features:
            features['pagination'] = features['pagination'] * ~missing[:, None]
        example['features'] = features
    mask_digest = hashlib.sha256(rows.astype('<i8').tobytes() + page_missing.tobytes()
                                 + modality_keep.tobytes()).hexdigest()
    return {'seed': seed, 'modality_probability': modality_probability, 'page_probability': page_probability,
            'applied': apply, 'mask_sha256': mask_digest, 'unique_pages': len(rows),
            'complete_missing_unique_pages': int(page_missing.sum()),
            'groups': 'explicit complete-page mask versus all other pages; boundary assigned to current page'}


def evaluate_robustness(model, dataset, blocks, window_size, output_dir, step, device='cpu',
                        batch_size=64, tracking=None):
    """Eight passes at the training context length, not a full context-length grid.

    Six main-modality subsets plus paired normal/corrupted inputs. Pagination is
    retained in the subset ablations; complete-page corruption removes it too.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        'image_missing': {'omit_modalities': ('image',)},
        'text_missing': {'omit_modalities': ('text',)},
        'stats_missing': {'omit_modalities': ('stats',)},
        'image_only': {'omit_modalities': ('text', 'stats')},
        'text_only': {'omit_modalities': ('image', 'stats')},
        'stats_only': {'omit_modalities': ('image', 'text')},
        'normal_paired': {'corruption': {'apply': False}},
        'scattered': {'corruption': {'apply': True}},
    }
    summary = {}
    for name, options in cases.items():
        report = evaluate_contexts(model, dataset, blocks, [window_size], device,
                                   batch_size=batch_size, **options)
        report.update(checkpoint_step=step, case=name,
                      pagination_in_subset_ablations='retained if available')
        (output_dir / f'final-robustness-{name}.json').write_text(json.dumps(report, allow_nan=False) + '\n')
        if tracking is not None:
            tracking.report_contexts(report, step, prefix=f'final_robustness/{name}')
            for group, scores in report['lengths'][str(window_size)].get('groups', {}).items():
                grouped = dict(report, lengths={str(window_size): dict(scores, complete=report['complete'])})
                tracking.report_contexts(grouped, step, prefix=f'final_robustness/{name}/{group}')
        summary[name] = {'complete': report['complete'], 'seconds': report['seconds'],
                         'panel_sha256': report['panel_sha256']}
    (output_dir / 'final-robustness-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    from .runner import evaluation_dataset, load_checkpoint
    from .manifests import load_streams
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Trusted local training checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--training-store')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample-pages', type=int, default=8192)
    args = parser.parse_args()
    model, state = load_checkpoint(args.checkpoint, args.device)
    config, dataset = evaluation_dataset(state, args.training_store)
    streams = load_streams(config['dev_manifest'], dataset.index, 'dev')
    blocks = context_blocks(dataset.index, streams, args.sample_pages)
    evaluate_robustness(model, dataset, blocks, config['window_size'], args.output_dir,
                        state['step'], args.device, config['inference']['batch_size'])


if __name__ == '__main__':
    main()
