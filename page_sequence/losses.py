"""Independent valid-target denominators. Missing supervision is never an extra class."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .schema import IGNORE


@dataclass(frozen=True)
class LossWeights:
    gold_type: float = 1.0
    weak_type: float = .2
    placement: float = .5
    starts: float = .5
    page_start: float = .25
    page_end: float = .25


def task_losses(output, batch, weights=LossWeights()):
    valid = batch['attention_mask']
    type_mask = valid & (batch['type'] != IGNORE)
    gold = type_mask & batch['gold']
    masks = {'gold_type': gold, 'weak_type': type_mask & ~batch['gold'],
             'placement': valid & (batch['placement'] != IGNORE),
             'starts': valid & batch['pair_mask'] & output['pair_mask']}
    for task in ('page_start', 'page_end'):
        if task + '_logits' in output:
            masks[task] = valid & (batch[task] != IGNORE)
    # Compute only selected targets: even a deliberately invalid masked label cannot affect loss.
    result, denominators = {}, {}
    for task, mask in masks.items():
        logits = output[task + '_logits' if task in ('page_start', 'page_end') else
                        'starts_document_logits' if task == 'starts' else
                        'placement_logits' if task == 'placement' else 'type_logits']
        target = batch[task if task in ('page_start', 'page_end', 'starts') else
                       'placement' if task == 'placement' else 'type']
        weight = batch['pair_weight' if task == 'starts' else 'weight'][mask]
        denom = weight.sum()
        denominators[task] = denom.detach()
        if mask.any():
            raw = (F.binary_cross_entropy_with_logits(logits[mask].float(), target[mask].float(), reduction='none')
                   if task in ('starts', 'page_start', 'page_end') else
                   F.cross_entropy(logits[mask].float(), target[mask], reduction='none'))
            result[task] = (raw * weight).sum() / denom.clamp_min(1e-12)
        else:
            result[task] = logits.float().sum() * 0
    result['total'] = sum(getattr(weights, k) * v for k, v in result.items())
    return result, denominators
