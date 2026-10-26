"""Build a completely synthetic vector/index fixture and exercise the real training runner.

Run artifacts must be outside the repository, e.g. /home/ikohut/tmp/page-sequence-smoke.
Nothing here reads page text or OCR. No downloads are required.
"""
import argparse
import json
from pathlib import Path

import lmdb
import numpy as np

from embeddings.text.page_stats import PageStats
from embeddings.text.page_stats.page_stats import to_sparse

from .index import build_index, PageIndex
from .manifests import write_streams
from .runner import train
from .schema import Page


def prepare_fixture(directory, steps=80):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(19)
    documents, pages = [], []
    for i in range(12):
        split = 'fit' if i < 8 else 'dev' if i < 10 else 'test'
        documents.append({'library': 'fixture', 'doc': f'd{i}', 'split': split,
                          'volume': f'{split}-v{i // 2}', 'periodical': f'{split}-periodical'})
        for j in range(10):
            label = 'TitlePage' if j == 0 else 'Blank' if j == 9 else 'NormalPage'
            pages.append(Page('fixture', f'd{i}', f'p{i}-{j}', j, f'public.256/d{i}/{j}.jpg',
                              label, label if j % 3 == 0 else '', 'left' if j % 2 else 'right'))
    build_index(directory / 'index.sqlite', documents, pages)
    for kind in ('image', 'text', 'stats'):
        env = lmdb.open(str(directory / kind), map_size=16 * 1024 * 1024)
        with env.begin(write=True) as txn:
            if kind == 'stats':
                txn.put(b'__metadata__', json.dumps(PageStats().metadata()).encode())
            for page in pages:
                if kind == 'stats':
                    vector = np.zeros(len(PageStats.features), np.float32)
                    vector[PageStats.features.index('log_chars')] = np.log1p(10 + page.order)
                    raw = to_sparse(vector)
                else:
                    vector = rng.normal(0, .01, 16).astype(np.float32)
                    vector[0 if page.order == 0 else 1 if page.order == 9 else 2] += 2
                    vector[3 + page.order % 2] += 2
                    raw = vector.astype('<f2').tobytes()
                key = page.path.split('/', 1)[1] if kind == 'image' else page.key
                txn.put(key.encode(), raw)
        env.close()
    index = PageIndex(directory / 'index.sqlite')
    write_streams(index, 'dev', directory / 'dev.jsonl', target_pages=20)
    write_streams(index, 'test', directory / 'test.jsonl', target_pages=20)
    config = {'seed': 17, 'device': 'cpu', 'deterministic': True,
              'index': str(directory / 'index.sqlite'), 'dev_manifest': str(directory / 'dev.jsonl'),
              'run_dir': str(directory / 'run'), 'features': {'sources': {
                  name: {'stores': {'public': str(directory / name)}, 'kind': name,
                         'dim': len(PageStats.features) if name == 'stats' else 16}
                  for name in ('image', 'text', 'stats')}},
              'backbone': {'kind': 'bert', 'pretrained': False, 'config': {
                  'vocab_size': 16, 'hidden_size': 32, 'num_hidden_layers': 2,
                  'num_attention_heads': 4, 'intermediate_size': 64, 'max_position_embeddings': 128,
                  'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.}},
              'fusion': 'linear', 'window_size': 12, 'batch_size': 4,
              'pretrain_steps': steps * 3 // 4, 'finetune_steps': steps - steps * 3 // 4,
              'validate_every': max(1, steps // 2),
              'optimizer': {'encoder_lr': .003, 'new_lr': .005, 'weight_decay': .01,
                            'warmup_fraction': .05, 'clip_grad': 1.},
              'inference': {'window_size': 12, 'stride': 6, 'merge': 'center_logits',
                            'edge_ramp': 3, 'start_threshold': .5, 'batch_size': 4}}
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.steps < 4:
        parser.error('--steps must be >=4')
    config = prepare_fixture(args.output_dir, args.steps)
    if not args.prepare_only:
        # Avoid many CPU threads dominating the tiny fixture's matrix operations.
        import torch
        torch.set_num_threads(2)
        print(train(config))


if __name__ == '__main__':
    main()
