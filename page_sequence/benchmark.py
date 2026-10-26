"""Bounded-memory measurement of the current metadata/vector reader and GPU training.

Consumes an existing index/config; does not modify vector stores or flush OS caches.
Only one optimizer batch is materialized at a time. Results describe the supplied sample.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .data import WindowDataset, collate_windows, to_device
from .features import FeaturePipeline
from .index import PageIndex
from .losses import task_losses
from .models import BertBackbone, PageTagger
from .schema import Window


def process_usage():
    io = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
    memory = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:]:
        name, _, value = line.partition(':')
        if name in ('Rss', 'Pss_Anon', 'Pss_File', 'Private_Dirty'):
            memory[name + '_KiB'] = int(value.split()[0])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'read_bytes': int(io['read_bytes']), 'major_faults': usage.ru_majflt,
            'peak_rss_KiB': usage.ru_maxrss, **memory}


def summarize(seconds, pages, before, after):
    total = sum(seconds)
    return {'seconds': total, 'pages': pages, 'pages_per_second': pages / total,
            'batch_seconds_p50': float(np.median(seconds)),
            'batch_seconds_p95': float(np.percentile(seconds, 95)),
            'process_disk_read_bytes': after['read_bytes'] - before['read_bytes'],
            'major_faults': after['major_faults'] - before['major_faults'],
            'memory_after': after}


def run(config, batches=64, seed=17, gpu=False):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    index = PageIndex(config['index'])
    pipeline = FeaturePipeline.from_config(config['features'])
    dataset = WindowDataset(index, pipeline)
    window_size, batch_size = config.get('window_size', 128), config.get('batch_size', 8)
    docs = [d for d in index.documents('fit') if d.length >= window_size]
    if not docs or batches < 1 or batch_size < 1:
        raise ValueError('positive batch budget and eligible documents required')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    windows = []
    for i in range(batches * batch_size):
        d = docs[int(order[i % len(order)])]
        start = d.first + int(rng.integers(d.length - window_size + 1))
        windows.append(Window(tuple(range(start, start + window_size)), 'benchmark'))
    report = {'sample': {'eligible_docs': len(docs), 'libraries': sorted({d.library for d in docs}),
        'indexed_pages': index.db.execute('SELECT COUNT(*) FROM pages').fetchone()[0],
        'unique_measured_pages': len({r for w in windows for r in w.rows}),
        'window_size': window_size, 'batch_size': batch_size, 'batches': batches,
        'seed': seed}, 'config': config,
        'limitations': ['OS caches are not flushed; first pass is not guaranteed cold.',
            'Windows are within documents, not the full training mixture or boundary sampler.',
            'Sampler catalogue memory on the full corpus is not measured by this sample.',
            'Only the configured stores and their current extraction state are measured.']}

    def emit(name, result):
        report[name] = result
        print(json.dumps({name: result}), flush=True)

    print(json.dumps({'sample': report['sample']}), flush=True)

    def load_batch(offset):
        return collate_windows([dataset[w] for w in windows[offset:offset + batch_size]])

    # Open environments/schema before timing, without touching sampled vector keys.
    for source in pipeline.sources.values():
        source._open()
    for phase in ('first_loading_pass', 'repeat_loading_pass'):
        before = process_usage()
        times, presence = [], np.zeros(3)
        for offset in range(0, len(windows), batch_size):
            started = time.perf_counter()
            batch = load_batch(offset)
            times.append(time.perf_counter() - started)
            presence += batch['features']['presence'].sum((0, 1)).numpy()
            if (offset // batch_size + 1) % 16 == 0:
                print(json.dumps({'phase': phase, 'batches_done': offset // batch_size + 1}), flush=True)
        pages = len(windows) * window_size
        result = summarize(times, pages, before, process_usage())
        result['presence_fraction'] = dict(zip(('image', 'text', 'stats'), (presence / pages).tolist()))
        emit(phase, result)

    profile_windows = windows[:min(32, len(windows))]
    before = process_usage()
    times = []
    for w in profile_windows:
        started = time.perf_counter()
        index.pages(w.rows)
        times.append(time.perf_counter() - started)
    emit('metadata_only', summarize(times, len(profile_windows) * window_size, before, process_usage()))
    pages = index.pages(tuple(r for w in profile_windows for r in w.rows))
    for name, source in pipeline.sources.items():
        before = process_usage()
        times = []
        for offset in range(0, len(pages), window_size):
            started = time.perf_counter()
            source.read(pages[offset:offset + window_size])
            times.append(time.perf_counter() - started)
        emit(name + '_decode_only', summarize(times, len(pages), before, process_usage()))
    if gpu:
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        backbone = dict(config['backbone'])
        backbone.pop('kind', None)
        model = PageTagger(BertBackbone.create(**backbone), pipeline.dims,
                           config.get('fusion', 'separate_mlp')).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        report['gpu'] = {'name': torch.cuda.get_device_name(),
                         'parameters': sum(p.numel() for p in model.parameters()), 'dtype': 'float32'}

        def update(batch):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['features'], batch['attention_mask'])
            loss = task_losses(output, batch)[0]['total']
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite benchmark loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            return float(loss.detach())

        cached_batch = to_device(load_batch(0), 'cuda')
        for _ in range(5):
            update(cached_batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = process_usage()
        times = []
        for _ in range(32):
            started = time.perf_counter()
            loss = update(cached_batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        result = summarize(times, 32 * batch_size * window_size, before, process_usage())
        result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['last_loss'] = loss
        emit('cached_batch_gpu_training', result)
        # Use different documents/windows for the end-to-end pass where the sample permits.
        measured_rows = {r for w in windows for r in w.rows}
        training_windows = []
        for i in range(len(windows)):
            d = docs[int(order[(len(windows) + i) % len(order)])]
            start = d.first + int(rng.integers(d.length - window_size + 1))
            training_windows.append(Window(tuple(range(start, start + window_size)), 'benchmark_training'))
        windows = training_windows
        training_rows = {r for w in windows for r in w.rows}
        report['training_sample'] = {'unique_pages': len(training_rows),
                                     'overlap_with_loading_sample': len(training_rows & measured_rows)}
        before = process_usage()
        times, loading, presence = [], [], np.zeros(3)
        for offset in range(0, len(windows), batch_size):
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = to_device(load_batch(offset), 'cuda')
            torch.cuda.synchronize()
            loading.append(time.perf_counter() - started)
            loss = update(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            presence += batch['features']['presence'].sum((0, 1)).cpu().numpy()
            if (offset // batch_size + 1) % 16 == 0:
                print(json.dumps({'phase': 'synchronous_training', 'batches_done': offset // batch_size + 1}), flush=True)
        result = summarize(times, len(windows) * window_size, before, process_usage())
        result['loading_and_transfer_fraction'] = sum(loading) / sum(times)
        result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['last_loss'] = loss
        result['presence_fraction'] = dict(zip(('image', 'text', 'stats'),
            (presence / (len(windows) * window_size)).tolist()))
        emit('synchronous_training', result)
    report['memory_final'] = process_usage()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    report = run(json.loads(Path(args.config).read_text()), args.batches, args.seed, args.gpu)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
