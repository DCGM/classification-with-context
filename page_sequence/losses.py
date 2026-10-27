"""Independent valid-target denominators. Missing supervision is never an extra class."""
from dataclasses import dataclass, asdict
import math

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
    boundary_consistency: float = 0.


def stage_loss_weights(config, stage):
    """Merge optional stage overrides; unknown tasks/stages fail before training."""
    if stage not in ('pretrain', 'finetune'):
        raise ValueError('unknown loss-weight stage')
    overrides = config.get('stage_loss_weights', {})
    if set(overrides) - {'pretrain', 'finetune'}:
        raise ValueError('unknown loss-weight stage override')
    values = dict(config.get('loss_weights', {}))
    values.update(overrides.get(stage, {}))
    try:
        weights = LossWeights(**values)
    except TypeError as error:
        raise ValueError('unknown loss-weight task') from error
    if any(not math.isfinite(value) or value < 0 for value in asdict(weights).values()):
        raise ValueError('loss weights must be finite and nonnegative')
    return weights


def bernoulli_js(a, b):
    """Symmetric Jensen-Shannon divergence of Bernoulli logits, in FP32."""
    log_a = torch.stack((F.logsigmoid(-a.float()), F.logsigmoid(a.float())), -1)
    log_b = torch.stack((F.logsigmoid(-b.float()), F.logsigmoid(b.float())), -1)
    log_mean = torch.logaddexp(log_a, log_b) - math.log(2)
    return (.5 * ((log_a.exp() * (log_a - log_mean)).sum(-1)
                  + (log_b.exp() * (log_b - log_mean)).sum(-1))).clamp_min(0)


def boundary_consistency(output, batch):
    """Agree on boundary(i), start(i), end(i-1); only coherent labelled pairs.

    No detach: gradients reach all three heads and their shared contextual encoder.
    Outer scan edges without a neighbour retain supervised auxiliary losses only.
    """
    if any(key not in output for key in ('page_start_logits', 'page_end_logits')):
        raise ValueError('boundary consistency requires auxiliary start/end heads')
    pair = batch['attention_mask'] & batch['pair_mask'] & output['pair_mask']
    mask = pair[:, 1:] & batch['attention_mask'][:, :-1]
    start, end = batch['page_start'][:, 1:], batch['page_end'][:, :-1]
    known = (start != IGNORE) & (end != IGNORE)
    coherent = (start == batch['starts'][:, 1:]) & (end == batch['starts'][:, 1:])
    conflicts = (mask & known & ~coherent).sum().detach()
    mask &= known & coherent
    weights = batch['pair_weight'][:, 1:][mask]
    denom = weights.sum()
    if mask.any():
        b = output['starts_document_logits'][:, 1:][mask]
        s = output['page_start_logits'][:, 1:][mask]
        e = output['page_end_logits'][:, :-1][mask]
        raw = .5 * (bernoulli_js(b, s) + bernoulli_js(b, e))
        loss = (raw * weights).sum() / denom.clamp_min(1e-12)
    else:
        loss = sum(output[key].float().sum() for key in
                   ('starts_document_logits', 'page_start_logits', 'page_end_logits')) * 0
    return loss, denom.detach(), conflicts


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
    if weights.boundary_consistency:
        loss, denom, conflicts = boundary_consistency(output, batch)
        result['boundary_consistency'] = loss
        denominators['boundary_consistency'] = denom
        denominators['consistency_conflicting_targets'] = conflicts
    result['total'] = sum(getattr(weights, k) * v for k, v in result.items())
    return result, denominators
