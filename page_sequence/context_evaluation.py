"""Bounded short-scan diagnostics, independent of checkpoint selection and backbone."""
import hashlib
import math
import time

import numpy as np
import torch

from .data import collate_windows, to_device
from .evaluation import binary_metrics, classification_metrics
from .precision import autocast
from .schema import IGNORE, PAGE_TYPES, PLACEMENTS, Window


def context_blocks(index, streams, sample_pages=2048, block_size=256, seed=917):
    """Fixed dev-only blocks; rotate curated, boundary and uniform anchors.

    Every length partitions the same blocks, so page coverage is identical when
    complete. Boundary supervision excludes each newly created scan's first page.
    This deliberately enriched small panel is a diagnostic, not a corpus estimate.
    """
    if sample_pages < 1 or block_size < 1 or not streams:
        raise ValueError('nonempty development panel required')
    if any(s['split'] != 'dev' for s in streams):
        raise ValueError('context diagnostics must use development streams only')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(streams))
    result, remaining = [], sample_pages
    for j in range(math.ceil(sample_pages / block_size)):
        stream = streams[int(order[j % len(order)])]
        pages = index.pages(stream['rows'])
        candidates = ([i for i, p in enumerate(pages) if p.target()[1]] if j % 3 == 0 else
                      [i for i in range(1, len(pages)) if pages[i].doc_key != pages[i - 1].doc_key]
                      if j % 3 == 1 else [])
        anchor = int(rng.choice(candidates)) if candidates else int(rng.integers(len(pages)))
        n = min(remaining, block_size, len(pages))
        start = max(0, min(anchor - n // 2, len(pages) - n))
        result.append(tuple(stream['rows'][start:start + n]))
        remaining -= n
        if not remaining:
            break
    return result


@torch.inference_mode()
def evaluate_contexts(model, dataset, blocks, lengths, device='cpu', batch_size=64, seconds_budget=None,
                      omit_pagination=False):
    """Batch independent scans at each length; stop between batches at the budget.

    A single in-flight batch may overrun the deadline. Partial levels explicitly
    report coverage and must not be compared as if they covered the whole panel.
    No training RNG is consumed and the model's previous train/eval state is restored.
    """
    if not blocks or batch_size < 1 or any(n < 1 for n in lengths):
        raise ValueError('nonempty blocks, positive lengths and batch size required')
    start = time.perf_counter()
    deadline = float('inf') if seconds_budget is None else start + max(0., seconds_budget)
    report = {'panel_pages': sum(map(len, blocks)), 'lengths': {}, 'seconds_budget': seconds_budget,
              'panel_sha256': hashlib.sha256(repr(blocks).encode()).hexdigest(),
              'description': 'fixed enriched dev blocks, same pages at each complete length; first-page boundary excluded'}
    # Read each panel block once. Re-reading thousands of singleton windows
    # would spend the diagnostic budget on storage and repeated record setup.
    examples = ([dataset[Window(block)] for block in blocks]
                if time.perf_counter() < deadline else [])
    if omit_pagination:
        for example in examples:
            if 'pagination' not in example['features']:
                raise ValueError('pagination ablation requires pagination features')
            example['features'] = dict(example['features'],
                                       pagination=np.zeros_like(example['features']['pagination']))
        report['pagination'] = 'omitted'
    was_training = model.training
    model.eval()
    try:
        for length in lengths:
            windows = []
            for example in examples:
                for lo in range(0, len(example['rows']), length):
                    hi = min(lo + length, len(example['rows']))
                    sliced = {key: example[key][lo:hi] for key in ('type', 'gold', 'placement', 'starts')}
                    sliced.update(features={k: v[lo:hi] for k, v in example['features'].items()},
                                  rows=example['rows'][lo:hi], source='context_diagnostic',
                                  pair_mask=np.r_[False, example['pair_mask'][lo + 1:hi]])
                    windows.append(sliced)
            cms = {name: np.zeros((n, n), np.int64) for name, n in (
                ('curated_type', len(PAGE_TYPES)), ('weak_type_agreement', len(PAGE_TYPES)),
                ('placement_db_agreement', len(PLACEMENTS)))}
            y_boundary, p_boundary = [], []
            pages_seen, scans_seen = 0, 0
            for first in range(0, len(windows), batch_size):
                if time.perf_counter() >= deadline:
                    break
                current = windows[first:first + batch_size]
                batch = to_device(collate_windows(current), device,
                                  feature_dtype=torch.float32 if getattr(model, 'compute_precision', 'float32')
                                  == 'float32' else None)
                with autocast(model, device):
                    output = model(batch['features'], batch['attention_mask'])
                valid = batch['attention_mask']
                for name, target, predicted, mask in (
                    ('curated_type', batch['type'], output['type_logits'].argmax(-1),
                     valid & (batch['type'] != IGNORE) & batch['gold']),
                    ('weak_type_agreement', batch['type'], output['type_logits'].argmax(-1),
                     valid & (batch['type'] != IGNORE) & ~batch['gold']),
                    ('placement_db_agreement', batch['placement'], output['placement_logits'].argmax(-1),
                     valid & (batch['placement'] != IGNORE))):
                    y, p = target[mask].cpu().numpy(), predicted[mask].cpu().numpy()
                    np.add.at(cms[name], (y, p), 1)
                pair_mask = valid & batch['pair_mask'] & output['pair_mask']
                y_boundary.extend(batch['starts'][pair_mask].cpu().tolist())
                p_boundary.extend(output['starts_document_logits'][pair_mask].float().sigmoid().cpu().tolist())
                pages_seen += sum(len(w['rows']) for w in current)
                scans_seen += len(current)
            report['lengths'][str(length)] = {
                'complete': bool(examples) and scans_seen == len(windows), 'pages': pages_seen, 'scans': scans_seen,
                **{name: classification_metrics(cm, PLACEMENTS if name.startswith('placement') else PAGE_TYPES)
                   for name, cm in cms.items()},
                'starts': binary_metrics(y_boundary, p_boundary, .5) if y_boundary else None}
            if time.perf_counter() >= deadline:
                break
    finally:
        model.train(was_training)
    report['seconds'] = time.perf_counter() - start
    report['complete'] = (len(report['lengths']) == len(lengths)
                          and all(r['complete'] for r in report['lengths'].values()))
    return report
