"""Partition sampled parent windows without changing their pages or augmentation."""
import torch

from .losses import task_losses
from .precision import autocast
from .schema import IGNORE


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


def slice_batch(batch, lo, hi):
    return {k: ({n: x[lo:hi] for n, x in v.items()} if k == 'features' else v[lo:hi])
            for k, v in batch.items()}


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


def backward_chunks(model, batch, features, size, microbatch, weights, scaler, device):
    """One optimizer update's loss, normalized over ALL chunks, not per chunk."""
    if microbatch < 1 or weights.boundary_consistency:
        raise ValueError('positive microbatch and no cross-chunk consistency required')
    chunks = split_context(dict(batch, features=features), size)
    denoms = target_denominators(chunks)
    totals = {}
    for lo in range(0, len(chunks['attention_mask']), microbatch):
        part = slice_batch(chunks, lo, lo + microbatch)
        with autocast(model, device):
            output = model(part['features'], part['attention_mask'])
        losses, _ = task_losses(output, part, weights, normalization_denominators=denoms)
        if not torch.isfinite(losses['total']):
            raise FloatingPointError('nonfinite training loss')
        scaler.scale(losses['total']).backward()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0) + value.detach()
    return totals, denoms
