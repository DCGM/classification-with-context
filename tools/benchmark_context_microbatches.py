"""Compare real optimizer updates with fixed parent data and physical chunk batches.

Runs outside ClearML and never writes training checkpoints. The caller pauses
other trainers and provides their frozen source through PYTHONPATH.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from page_sequence.augmentation import pagination_dropout, modality_dropout, scattered_dropout
from page_sequence.context_training import backward_chunks
from page_sequence.data import to_device
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.loading import training_loader, release_host_cache
from page_sequence.losses import stage_loss_weights
from page_sequence.runner import load_checkpoint, make_dataset, make_optimizer, seed_all
from page_sequence.scan_sampling import ScanBatchSampler


def describe(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values),
                p90=ordered[min(len(ordered) - 1, int(.9 * len(ordered)))])


def benchmark(config_path, output, loading_updates=12, loading_warmup=2, loading_only=False):
    config = json.loads(config_path.read_text())
    if config['window_size'] != 1:
        raise ValueError('this benchmark requires a W1 training recipe')
    torch.set_num_threads(config.get('cpu_threads', 1))
    checkpoint = Path(config['run_dir']) / 'latest.pt'
    model, state = load_checkpoint(checkpoint, 'cuda')
    model.train()
    optimizer = make_optimizer(model, config['optimizer'])
    stage = state['training']['stage']
    weights = stage_loss_weights(config, stage)
    index = PageIndex(config['index'])
    dataset = make_dataset(index, FeaturePipeline.from_config(config['features']), config)
    sampler = ScanBatchSampler(index, config['sampling_window_size'], config['seed'], stage=stage,
                               variable_lengths=config['variable_lengths'],
                               scan_batches=config['scan_batches'])

    def loader_for(count):
        sampler.load_state_dict(state['sampler'])
        loader, batches = training_loader(dataset, sampler, config)
        batches.batches = count
        return loader

    def close_loader(loader):
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
        gc.collect()
        release_host_cache('cuda')

    def reset():
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        seed_all(config['seed'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        return scaler

    def update(raw, microbatch, scaler):
        batch = to_device(raw, 'cuda')
        features = pagination_dropout(batch['features'], **config['pagination_dropout'])
        features = modality_dropout(features, **config['modality_dropout'])
        features = scattered_dropout(features, **config['scattered_dropout'])
        optimizer.zero_grad(set_to_none=True)
        losses, _ = backward_chunks(model, batch, features, 1, microbatch, weights, scaler, 'cuda')
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad'])
        if not torch.isfinite(norm):
            raise FloatingPointError('nonfinite benchmark gradients')
        scaler.step(optimizer)
        scaler.update()
        loss = float(losses['total'])
        if not __import__('math').isfinite(loss):
            raise FloatingPointError('nonfinite benchmark loss')
        return loss

    report = dict(config=str(config_path), checkpoint=str(checkpoint), checkpoint_step=state['step'],
                  stage=stage, device=torch.cuda.get_device_name(), warmup_updates=2,
                  measured_cached_updates=0 if loading_only else 8,
                  measured_loading_updates=loading_updates, loading_warmup_updates=loading_warmup,
                  parent_batch=64, parent_window=512, context=1, workers=config['workers'],
                  cached=[], end_to_end=[])

    def persist():
        output.write_text(json.dumps(report, indent=2) + '\n')

    loader = loader_for(4)
    cached = list(loader)
    close_loader(loader)
    report['sample_pages'] = [int(x['attention_mask'].sum()) for x in cached]
    report['sample_rows_sha256'] = hashlib.sha256(repr([x['rows'] for x in cached]).encode()).hexdigest()
    assert all(x['storage_pages']['fallback'] == 0 for x in cached)
    persist()
    for microbatch in ([] if loading_only else [1024, 2048, 4096, 8192, 16384, 32768]):
        scaler = reset()
        times, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(10):
                raw = cached[i % len(cached)]
                torch.cuda.synchronize()
                start = time.perf_counter()
                loss = update(raw, microbatch, scaler)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if i >= 2:
                    times.append(elapsed); losses.append(loss)
            result = dict(physical_batch=microbatch, seconds=describe(times),
                          losses=losses, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            result = dict(physical_batch=microbatch, oom=True)
        report['cached'].append(result)
        persist(); print(json.dumps(dict(mode='cached', **result)), flush=True)
    fastest = (min((r for r in report['cached'] if not r.get('oom')),
                   key=lambda r: r['seconds']['median'])['physical_batch']
               if report['cached'] else 32768)
    del cached
    gc.collect()
    report['fastest_cached_physical_batch'] = fastest
    # Compare current policy with the best measured choice under normal 8-worker loading.
    for microbatch in dict.fromkeys([1024, fastest]):
        scaler = reset()
        loader = loader_for(loading_warmup + loading_updates)
        iterator = iter(loader)
        times, io, pages, rows = [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        for i in range(loading_warmup + loading_updates):
            torch.cuda.synchronize()
            start = time.perf_counter()
            raw = next(iterator)
            loaded = time.perf_counter() - start
            update(raw, microbatch, scaler)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if i >= loading_warmup:
                times.append(elapsed); io.append(loaded)
                pages.append(int(raw['attention_mask'].sum())); rows.append(raw['rows'])
        try:
            next(iterator)
            raise RuntimeError('unexpected extra loader batch')
        except StopIteration:
            pass
        result = dict(physical_batch=microbatch, seconds=describe(times), loading_seconds=describe(io),
                      pages=sum(pages), pages_per_second=sum(pages) / sum(times),
                      rows_sha256=hashlib.sha256(repr(rows).encode()).hexdigest(),
                      peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
        report['end_to_end'].append(result)
        close_loader(loader)
        persist(); print(json.dumps(dict(mode='end_to_end', **result)), flush=True)
    assert len({r['rows_sha256'] for r in report['end_to_end']}) == 1
    report['complete'] = True
    persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loading-updates', type=int, default=12)
    parser.add_argument('--loading-warmup', type=int, default=2)
    parser.add_argument('--loading-only', action='store_true')
    args = parser.parse_args()
    benchmark(args.config, args.output, args.loading_updates, args.loading_warmup, args.loading_only)
