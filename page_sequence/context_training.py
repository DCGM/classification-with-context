"""Partition sampled parent windows without changing their pages or augmentation."""
from collections import Counter

import torch

from .losses import task_losses
from .precision import autocast
from .schema import IGNORE
from .inference import window_spans

TARGETS = ('type', 'placement', 'page_start', 'page_end')


def split_context(batch, size):
    """Preserve parent order, duplicate weights, targets and genuine border flags."""
    b, length = batch['attention_mask'].shape
    if size < 1:
        raise ValueError('positive context size required')
    padded = ((length + size - 1) // size) * size

    def partition(x, fill=0):
        if padded != length:
            extra = x.new_full((b, padded - length, *x.shape[2:]), fill)
            x = torch.cat((x, extra), dim=1)
        return x.reshape(b * (padded // size), size, *x.shape[2:])

    result = {}
    for key, value in batch.items():
        if key == 'features':
            result[key] = {name: partition(x) for name, x in value.items()}
        elif isinstance(value, torch.Tensor):
            result[key] = partition(value, IGNORE if key in
                                    ('type', 'placement', 'page_start', 'page_end') else 0)
    keep = result['attention_mask'].any(1)
    result = {k: ({n: x[keep] for n, x in v.items()} if k == 'features' else v[keep])
              for k, v in result.items()}
    result['pair_mask'][:, 0] = False
    result['pair_weight'][:, 0] = 0
    return result


def piece_spans(length, size, coverage, scan_end, parent_stride, coverage_stride):
    """Spans of one sampled parent window, in its own positions.

    Anchor windows are cut into consecutive pieces. A coverage window is one tile of a whole scan, tiled by the
    sampler at `parent_stride` with the regular tail; the scan itself is cut with `size` at `coverage_stride`
    (regular tail). Each scan window is assigned to the tile in which it starts, so the tiles of one scan together
    hold exactly that scanning, and every window lies inside its tile.
    """
    if not coverage:
        return [(lo, min(lo + size, length)) for lo in range(0, length, size)]
    if not scan_end:
        return [(lo, lo + size) for lo in range(0, parent_stride, coverage_stride)]
    return window_spans(length, size, coverage_stride, 'regular')


def scan_context(batch, size, parent_stride, coverage_stride):
    """Cut every parent into its pieces, keep targets and boundary features, and reweight repeated pages.

    Repeated pages and pairs get reciprocal weight over all pieces of the update, as `collate_windows` does
    for parent windows. The first position of a piece has no predecessor and no pair target.
    """
    lengths = batch['attention_mask'].sum(1).tolist()
    pieces = []
    for b, (length, source, scan_end) in enumerate(zip(lengths, batch['sources'], batch['scan_ends'])):
        spans = piece_spans(length, size, source == 'coverage', scan_end, parent_stride, coverage_stride)
        if any(lo < 0 or hi > length or hi <= lo for lo, hi in spans):
            raise ValueError('scan piece outside its parent window')
        pieces += [(b, lo, hi) for lo, hi in spans]
    count = len(pieces)
    device = batch['attention_mask'].device
    parent = torch.tensor([b for b, _, _ in pieces], device=device)
    offset = torch.arange(size, device=device)
    position = torch.tensor([lo for _, lo, _ in pieces], device=device)[:, None] + offset
    valid = offset < torch.tensor([hi - lo for _, lo, hi in pieces], device=device)[:, None]
    position = torch.where(valid, position, 0)

    def gather(x, fill):
        x = x[parent[:, None], position]
        mask = valid.reshape(*valid.shape, *([1] * (x.dim() - 2)))
        return torch.where(mask, x, torch.as_tensor(fill, dtype=x.dtype, device=device))

    result = {'features': {n: gather(x, 0) for n, x in batch['features'].items()}}
    for key, value in batch.items():
        if key != 'features' and isinstance(value, torch.Tensor):
            result[key] = gather(value, IGNORE if key in TARGETS else 0)
    result['attention_mask'] &= valid
    rows = [batch['rows'][b][lo:hi] for b, lo, hi in pieces]
    occurrences = Counter(r for piece in rows for r in piece)
    pairs = Counter(p for piece in rows for p in zip(piece, piece[1:]))
    weight = torch.zeros(count, size)
    pair_weight = torch.zeros(count, size)
    for i, piece in enumerate(rows):
        weight[i, :len(piece)] = torch.tensor([1 / occurrences[r] for r in piece])
        if len(piece) > 1:
            pair_weight[i, 1:len(piece)] = torch.tensor([1 / pairs[p] for p in zip(piece, piece[1:])])
    result['weight'] = weight.to(device)
    result['pair_weight'] = pair_weight.to(device)
    result['pair_mask'][:, 0] = False
    result['pair_weight'][:, 0] = 0
    result['rows'] = rows
    return result


def slice_batch(batch, lo, hi):
    return {k: ({n: x[lo:hi] for n, x in v.items()} if k == 'features' else v[lo:hi])
            for k, v in batch.items() if k == 'features' or isinstance(v, torch.Tensor)}


def target_denominators(batch):
    valid = batch['attention_mask']
    types = valid & (batch['type'] != IGNORE)
    masks = {'gold_type': types & batch['gold'], 'weak_type': types & ~batch['gold'],
             'placement': valid & (batch['placement'] != IGNORE),
             'starts': valid & batch['pair_mask']}
    for key in ('page_start', 'page_end'):
        masks[key] = valid & (batch[key] != IGNORE)
    return {k: batch['pair_weight' if k == 'starts' else 'weight'][m].sum().detach()
            for k, m in masks.items()}


def backward_chunks(model, batch, features, size, microbatch, weights, scaler, device, coverage_scan=None):
    """One optimizer update's loss, normalized over ALL chunks, not per chunk.

    `microbatch` limits the page positions of one physical batch. `coverage_scan`, a dict with the sampler's
    `parent_stride` and the `coverage_stride`, switches to `scan_context`; otherwise parents are cut into
    consecutive chunks (`split_context`).
    """
    if microbatch < 1 or weights.boundary_consistency:
        raise ValueError('positive microbatch and no cross-chunk consistency required')
    chunks = (split_context(dict(batch, features=features), size) if coverage_scan is None
              else scan_context(dict(batch, features=features), size, **coverage_scan))
    denoms = target_denominators(chunks)
    totals = {}
    per_batch = max(1, microbatch // size)
    for lo in range(0, len(chunks['attention_mask']), per_batch):
        part = slice_batch(chunks, lo, lo + per_batch)
        with autocast(model, device):
            output = model(part['features'], part['attention_mask'])
        losses, _ = task_losses(output, part, weights, normalization_denominators=denoms)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('nonfinite training loss')
        scaler.scale(losses['total']).backward()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0) + value.detach()
    return totals, denoms
