"""Training-only feature augmentation, independent of storage and model architecture."""
import math

import torch


def modality_dropout(features, all_probability=1., two_probability=0., single_probability=0.):
    """Sample whole-window subsets of image/text/stats; pagination is independent.

    Subsets of each size are equiprobable. Preserve an originally available signal
    on individual pages if natural missingness would otherwise leave none. Never
    invent a signal for pages with all three originally missing. No inverted scaling.
    """
    probabilities = (all_probability, two_probability, single_probability)
    if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-8)):
        raise ValueError('modality dropout probabilities must be finite, in [0, 1], and sum to 1')
    if not two_probability and not single_probability:
        return features
    flags = features['presence']
    if flags.shape[0] == 0:
        return features
    probabilities = flags.new_tensor([all_probability] + [two_probability / 3] * 3
                                    + [single_probability / 3] * 3, dtype=torch.float32)
    patterns = flags.new_tensor([[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    choices = torch.multinomial(probabilities, flags.shape[0], replacement=True)
    keep = patterns[choices, None, :].expand_as(flags).clone()
    lost = ((flags * keep).sum(-1) == 0) & (flags.sum(-1) > 0)
    rescue = torch.nn.functional.one_hot(flags.argmax(-1), 3).to(flags.dtype)
    keep = torch.where(lost.unsqueeze(-1), rescue, keep)
    result = dict(features, presence=flags * keep)
    for i, name in enumerate(('image', 'text', 'stats')):
        result[name] = features[name] * keep[..., i:i + 1]
    return result


def pagination_dropout(features, window_probability=0., page_probability=0.):
    """Return features with entire pagination vectors missing, without inverted scaling.

    Windows are dropped first; page dropout applies independently in retained windows.
    Zero probabilities consume no RNG. Torch's checkpointed RNG makes resume reproducible.
    Input tensors are never mutated; targets and other modalities remain unchanged.
    """
    probabilities = (window_probability, page_probability)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('pagination dropout probabilities must be finite and in [0, 1]')
    if not any(probabilities):
        return features
    if 'pagination' not in features:
        raise ValueError('pagination dropout requires pagination features')
    x = features['pagination']
    keep = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
    if window_probability:
        keep &= torch.rand((x.shape[0], 1), device=x.device) >= window_probability
    if page_probability:
        keep &= torch.rand(x.shape[:-1], device=x.device) >= page_probability
    return dict(features, pagination=x * keep.unsqueeze(-1))
