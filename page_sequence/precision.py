"""FP16 compute with FP32 master weights, optimizer and losses."""
from contextlib import nullcontext

import torch


def autocast(model, device):
    if getattr(model, 'compute_precision', 'float32') == 'float16':
        if torch.device(device).type != 'cuda':
            raise ValueError('float16 compute requires CUDA')
        return torch.autocast('cuda', dtype=torch.float16)
    return nullcontext()
