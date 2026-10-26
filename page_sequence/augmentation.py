"""Training-only feature augmentation, independent of storage and model architecture."""
import math

import torch


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
