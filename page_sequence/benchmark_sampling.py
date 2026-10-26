"""Measure online training-window sampling and SSD loading without warming the working set."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .benchmark_workers import memory, worker_init
from .data import WindowDataset, collate_windows, to_device
from .features import FeaturePipeline
from .index import PageIndex
from .losses import task_losses
from .models import BertBackbone, PageTagger
from .sampling import WindowSampler


class OnlineBatches:
    """Keep all sampling in the parent process; workers only materialize supplied windows."""
    def __init__(self, sampler, batches, batch_size):
        self.sampler, self.batches, self.batch_size = sampler, batches, batch_size
        self.seconds = 0.

    def __iter__(self):
        for _ in range(self.batches):
            start = time.perf_counter()
            windows = [self.sampler.draw() for _ in range(self.batch_size)]
            self.seconds += time.perf_counter() - start
            yield windows

    def __len__(self):
        return self.batches


def run(config, workers=8, batches=512, seed=391):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    vector_path = next(iter(features.sources['image'].stores.values()))
    device = Path(subprocess.check_output(
        ['findmnt', '-n', '-o', 'SOURCE', '--target', vector_path], text=True).strip()).name
    disk_stat = Path('/sys/class/block') / device / 'stat'

    def disk_read_bytes():
        return int(disk_stat.read_text().split()[2]) * 512 if disk_stat.exists() else None
    dataset = WindowDataset(index, features)
    backbone = dict(config['backbone'])
    backbone.pop('kind', None)
    model = PageTagger(BertBackbone.create(**backbone), features.dims,
                       config.get('fusion', 'separate_mlp')).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    size, batch_size = config.get('window_size', 128), config.get('batch_size', 8)
    warmup = config.get('benchmark_warmup_batches', 8)
    if warmup < 1:
        raise ValueError('positive benchmark warmup required')
    setup_start = time.perf_counter()
    sampler = WindowSampler(index, size, seed=seed)
    setup_seconds = time.perf_counter() - setup_start
    report = {'config': config, 'workers': workers, 'prefetch_factor': 1,
        'batches_per_stage': batches, 'warmup_batches': warmup, 'seed': seed,
        'sampler_setup_seconds': setup_seconds, 'indexed_documents': len(sampler.docs),
        'indexed_pages': index.db.execute('SELECT COUNT(*) FROM pages').fetchone()[0],
        'eligible_same_volume_groups': len(sampler.hard_volumes),
        'indexed_gold_pages': index.db.execute("SELECT COUNT(*) FROM pages WHERE annotated_type!=''").fetchone()[0],
        'vector_block_device': device,
        'note': 'Actual online WindowSampler; no vector working-set warmup or OS cache flushing. '
                'Public training subset; pretrain then finetune naturally share cache. '
                'No checkpoint is saved; full-corpus catalogue memory remains unmeasured.'}
    print(json.dumps({'sample': {k:v for k,v in report.items() if k!='config'}}), flush=True)

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch['features'], batch['attention_mask'])
        loss = task_losses(output, batch)[0]['total']
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite benchmark loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()

    options = {'num_workers': workers, 'pin_memory': True}
    if workers:
        options.update(prefetch_factor=1, persistent_workers=True,
                       multiprocessing_context='spawn', worker_init_fn=worker_init)
    for stage in ('pretrain', 'finetune'):
        sampler.set_stage(stage)
        sampler.counters.clear()
        sampler.seen_pairs.clear()
        sampler.seen_volumes.clear()
        source = OnlineBatches(sampler, batches + warmup, batch_size)
        loader = DataLoader(dataset, batch_sampler=source, collate_fn=collate_windows, **options)
        startup = time.perf_counter()
        iterator = iter(loader)
        for _ in range(warmup):
            update(to_device(next(iterator), 'cuda'))
        torch.cuda.synchronize()
        startup = time.perf_counter() - startup
        pids = [os.getpid()] + ([w.pid for w in iterator._workers] if workers else [])
        before = memory(pids)
        disk_before = disk_read_bytes()
        sampler_before = source.seconds
        seconds, waits, page_counts, presence = [], [], [], np.zeros(3)
        unique_rows = set()
        pages = 0
        block_start = started = time.perf_counter()
        block_pages = 0
        blocks = []
        for i in range(batches):
            tick = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - tick)
            update(to_device(batch, 'cuda'))
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - tick)
            n = int(batch['attention_mask'].sum())
            page_counts.append(n)
            pages += n
            block_pages += n
            presence += batch['features']['presence'].sum((0, 1)).numpy()
            for rows in batch['rows']:
                unique_rows.update(rows)
            if (i + 1) % 64 == 0:
                now = time.perf_counter()
                block = {'stage': stage, 'batches_done': i + 1,
                         'pages_per_second': block_pages / (now - block_start)}
                blocks.append(block)
                print(json.dumps(block), flush=True)
                block_start, block_pages = now, 0
        elapsed = time.perf_counter() - started
        disk_after = disk_read_bytes()
        after = memory(pids)
        result = {'seconds': elapsed, 'pages': pages, 'unique_pages': len(unique_rows),
            'pages_per_second': pages / elapsed,
            'waiting_for_loader_and_sampling_fraction': sum(waits) / sum(seconds),
            'main_sampling_seconds': source.seconds - sampler_before,
            'batch_seconds_p50': float(np.median(seconds)),
            'batch_seconds_p95': float(np.percentile(seconds, 95)),
            'startup_and_warmup_seconds': startup,
            'process_disk_read_bytes': after['read_bytes'] - before['read_bytes'],
            'vector_device_read_bytes': disk_after - disk_before if disk_before is not None else None,
            'memory_before': before, 'memory_after': after,
            'presence_fraction': dict(zip(('image', 'text', 'stats'), (presence / pages).tolist())),
            'sampler_counters_including_warmup': dict(sampler.counters), 'blocks': blocks}
        keep = max(1, len(seconds) - workers)
        result['steady_pages_per_second_excluding_prefetch_drain'] = sum(page_counts[:keep]) / sum(seconds[:keep])
        result['steady_wait_fraction_excluding_prefetch_drain'] = sum(waits[:keep]) / sum(seconds[:keep])
        report[stage] = result
        print(json.dumps({stage: result}), flush=True)
        if workers:
            iterator._shutdown_workers()
        del iterator, loader
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batches', type=int, default=512)
    args = parser.parse_args()
    if args.workers < 0 or args.batches < 1:
        parser.error('nonnegative workers and positive batches required')
    report = run(json.loads(Path(args.config).read_text()), args.workers, args.batches)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
