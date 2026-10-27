"""Bounded worker loading; exhaust each segment before saving sampler state."""
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader

from .data import collate_windows


def worker_init(_):
    torch.set_num_threads(1)


def release_host_cache(device):
    """Release idle pinned batches at an exhausted loader segment, never live tensors.

    CUDA's host allocator otherwise retains the prefetch high-water mark throughout
    validation. This is CPU RAM; torch.cuda.empty_cache() does not release it.
    Older Torch versions without this public API retain their original behavior.
    """
    if torch.device(device).type != 'cuda':
        return {}
    empty = getattr(getattr(torch, 'accelerator', None), 'empty_host_cache', None)
    stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
    if empty is None:
        return {}
    before = stats() if stats else {}
    empty()
    after = stats() if stats else {}
    return {'reserved_before': before.get('allocated_bytes.current'),
            'reserved_after': after.get('allocated_bytes.current'),
            'active_after': after.get('active_bytes.current')}


class TrainingBatches:
    def __init__(self, sampler, batch_size):
        self.sampler, self.batch_size, self.batches = sampler, batch_size, 0

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            yield [self.sampler.draw() for _ in range(self.batch_size)]


def training_loader(dataset, sampler, config):
    workers = config.get('workers', 0)
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    batches = TrainingBatches(sampler, config['batch_size'])
    options = dict(num_workers=workers, pin_memory=config['device'].startswith('cuda'),
                   generator=torch.Generator().manual_seed(config['seed']))
    if workers:
        method = config.get('worker_start_method', 'spawn')
        if method not in ('spawn', 'forkserver'):
            raise ValueError('worker_start_method must be spawn or forkserver; CUDA parents must not fork')
        if method == 'forkserver':
            # A separate CPU-only server shares imported code/pages across workers.
            # It never inherits the trainer's CUDA context or million-document sampler.
            mp.set_forkserver_preload(['torch', 'numpy', 'page_sequence.runner', 'training_store.dataset'])
        options.update(prefetch_factor=1, persistent_workers=True,
                       multiprocessing_context=method, worker_init_fn=worker_init)
    return DataLoader(dataset, batch_sampler=batches, collate_fn=collate_windows, **options), batches
