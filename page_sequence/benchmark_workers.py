"""Compare synchronous loading with bounded DataLoader worker prefetch on warmed pages."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, collate_windows, to_device
from .features import FeaturePipeline
from .index import PageIndex
from .losses import task_losses
from .models import BertBackbone, PageTagger
from .schema import Window


def worker_init(_):
    torch.set_num_threads(1)


def memory(pids):
    total = {'pss_KiB': 0, 'anonymous_pss_KiB': 0, 'read_bytes': 0}
    for pid in pids:
        fields = {}
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()[1:]:
            key, _, value = line.partition(':')
            fields[key] = int(value.split()[0])
        total['pss_KiB'] += fields['Pss']
        total['anonymous_pss_KiB'] += fields.get('Pss_Anon', 0)
        io = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
        total['read_bytes'] += int(io['read_bytes'])
    return total


def run(config, workers=8, batches=128, seed=29):
    import os
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = WindowDataset(index, features)
    size, batch_size, warmup = config.get('window_size', 128), config.get('batch_size', 8), 8
    docs = [d for d in index.documents('fit') if d.length >= size]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    windows = []
    for i in range((batches + warmup) * batch_size):
        d = docs[int(order[i % len(docs)])]
        first = d.first + int(rng.integers(d.length - size + 1))
        windows.append(Window(tuple(range(first, first + size))))
    groups = [windows[i:i + batch_size] for i in range(0, len(windows), batch_size)]
    # Both loaders see the same warmed working set. Do not retain its batches in memory.
    for i, group in enumerate(groups):
        collate_windows([dataset[w] for w in group])
        if (i + 1) % 32 == 0:
            print(json.dumps({'warming_batches': i + 1}), flush=True)
    backbone = dict(config['backbone'])
    backbone.pop('kind', None)
    model = PageTagger(BertBackbone.create(**backbone), features.dims,
                       config.get('fusion', 'separate_mlp')).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch['features'], batch['attention_mask'])
        loss = task_losses(output, batch)[0]['total']
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()

    sample = to_device(collate_windows([dataset[w] for w in groups[0]]), 'cuda')
    for _ in range(5):
        update(sample)
    torch.cuda.synchronize()
    report = {'config': config, 'batches': batches, 'warmup_batches': warmup,
        'window_size': size, 'batch_size': batch_size,
        'unique_pages': len({r for w in windows for r in w.rows}),
        'cpu_threads_main': 2, 'cpu_threads_per_worker': 1,
        'note': 'Same warmed pages, no cache flushing; full-corpus sampler memory is not measured.'}
    for count in (0, workers):
        options = {'num_workers': count, 'pin_memory': True}
        if count:
            options.update(prefetch_factor=1, persistent_workers=True,
                           multiprocessing_context='spawn', worker_init_fn=worker_init)
        loader = DataLoader(dataset, batch_sampler=groups, collate_fn=collate_windows, **options)
        startup = time.perf_counter()
        iterator = iter(loader)
        for _ in range(warmup):
            update(to_device(next(iterator), 'cuda'))
        torch.cuda.synchronize()
        startup = time.perf_counter() - startup
        pids = [os.getpid()] + ([w.pid for w in iterator._workers] if count else [])
        before = memory(pids)
        waits, seconds, presence = [], [], np.zeros(3)
        peak_memory = before
        started = time.perf_counter()
        for i in range(batches):
            tick = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - tick)
            gpu_batch = to_device(batch, 'cuda')
            update(gpu_batch)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - tick)
            presence += batch['features']['presence'].sum((0, 1)).numpy()
            if (i + 1) % 16 == 0:
                print(json.dumps({'workers': count, 'batches_done': i + 1}), flush=True)
        elapsed = time.perf_counter() - started
        after = memory(pids)
        if after['pss_KiB'] > peak_memory['pss_KiB']:
            peak_memory = after
        result = {'workers': count, 'prefetch_batches_per_worker': 1 if count else 0,
            'seconds': elapsed, 'pages_per_second': batches * batch_size * size / elapsed,
            'batch_seconds_p50': float(np.median(seconds)),
            'batch_seconds_p95': float(np.percentile(seconds, 95)),
            'waiting_for_loader_fraction': sum(waits) / sum(seconds),
            'startup_and_warmup_seconds': startup,
            'sampled_peak_parent_plus_workers_memory': peak_memory,
            'parent_plus_workers_disk_read_bytes': after['read_bytes'] - before['read_bytes'],
            'presence_fraction': dict(zip(('image', 'text', 'stats'),
                (presence / (batches * batch_size * size)).tolist()))}
        report[str(count)] = result
        print(json.dumps(result), flush=True)
        if count:
            iterator._shutdown_workers()
        del iterator, loader
    report['speedup'] = report[str(workers)]['pages_per_second'] / report['0']['pages_per_second']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batches', type=int, default=128)
    args = parser.parse_args()
    if args.workers < 1 or args.batches < 1:
        parser.error('workers and batches must be positive')
    report = run(json.loads(Path(args.config).read_text()), args.workers, args.batches)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
