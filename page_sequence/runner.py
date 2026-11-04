"""Small PyTorch runner; data, models, losses and full-stream evaluation remain reusable."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .data import WindowDataset, collate_windows, to_device
from .evaluation import Evaluator
from .context_evaluation import context_blocks, evaluate_contexts
from .features import FeaturePipeline
from .index import PageIndex
from .inference import InferenceConfig, predict_stream, tag_pages
from .losses import stage_loss_weights, task_losses
from .loading import release_host_cache, training_loader
from .tracking import Tracking
from .manifests import file_sha256, load_streams
from .models import BertBackbone, PageBackbone, PageTagger, from_specification
from .sampling import WindowSampler
from .schema import PAGE_TYPES, PLACEMENTS, Page
from .precision import autocast
from .augmentation import modality_dropout, pagination_dropout, scattered_dropout
from .modalities import ModalitySubsetDataset
from .context_training import backward_chunks


def make_dataset(index, features, config):
    if config.get('scan_borders') and config.get('boundary_features'):
        raise ValueError('choose boundary features or scan tokens, not both')
    if config.get('training_store'):
        from training_store.dataset import StoreDataset
        dataset = StoreDataset(index, features, **config['training_store'])
    else:
        dataset = WindowDataset(index, features)
    if config.get('pagination'):
        from .pagination import PaginationDataset
        dataset = PaginationDataset(dataset, **config['pagination'])
    if config.get('scan_borders'):
        from .scan_borders import ScanBorderDataset
        dataset = ScanBorderDataset(dataset)
    if config.get('boundary_features'):
        from .scan_borders import BoundaryFeatureDataset
        dataset = BoundaryFeatureDataset(dataset)
    if 'enabled_modalities' in config:
        dataset = ModalitySubsetDataset(dataset, config['enabled_modalities'])
    return dataset


def evaluation_dataset(state, training_store=None):
    """Relocate a published native store by identity, without consulting old LMDBs."""
    config = dict(state['config'])
    native = state['provenance'].get('training_store')
    if training_store is not None:
        if native is None:
            raise ValueError('checkpoint was not trained with a native training store')
        config['index'] = str(Path(training_store) / 'catalogue.sqlite')
        config['training_store'] = {'path': str(training_store), 'allow_building': False}
    elif native and (Path(config['training_store']['path']) / 'manifest.json').exists():
        config['training_store'] = dict(config['training_store'], allow_building=False)
    index = PageIndex(config['index'])
    if index.fingerprint() != state['provenance']['index_sha256']:
        raise ValueError('checkpoint/index mismatch')
    features = FeaturePipeline.from_config(config['features'])
    if features.state_dict() != state['features']:
        raise ValueError('checkpoint/feature schema mismatch')
    dataset = make_dataset(index, features, config)
    if native and not config['training_store'].get('allow_building', False):
        actual = dataset.source_provenance()
        if any(actual[k] != native[k] for k in ('build_id', 'plan_sha256')):
            raise ValueError('checkpoint/training-store identity mismatch')
    elif features.source_provenance() != state['provenance']['vector_sources']:
        raise ValueError('vector stores changed since training')
    if config.get('pagination') and dataset.pagination_provenance() != state['provenance'].get('pagination'):
        raise ValueError('pagination sidecar changed since training')
    return config, dataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, model, optimizer, sampler, step, best, config, provenance, feature_state, training_state=None):
    path = Path(path)
    temporary = path.with_suffix('.writing')
    torch.save({'format': 1, 'specification': model.specification(), 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'sampler': sampler.state_dict(), 'step': step,
                'best': best, 'rng': rng_state(), 'config': config, 'provenance': provenance,
                'features': feature_state, 'training': training_state,
                'grad_scaler': model.grad_scaler.state_dict() if hasattr(model, 'grad_scaler') else None,
                'page_types': PAGE_TYPES, 'placements': PLACEMENTS}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device='cpu'):
    # Only load trusted, locally produced checkpoints: training RNG/optimizer state uses pickle.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['format'] != 1 or tuple(state['page_types']) != PAGE_TYPES or tuple(state['placements']) != PLACEMENTS:
        raise ValueError('checkpoint schema/label mismatch')
    model = from_specification(state['specification'])
    model.load_state_dict(state['model'], strict=True)
    model.compute_precision = (state['config'].get('precision', 'float32')
                               if torch.device(device).type == 'cuda' else 'float32')
    return model.to(device), state


def make_optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder = name.startswith('backbone.')
        decay = parameter.ndim >= 2 and not name.endswith('bias')
        groups.setdefault((encoder, decay), []).append(parameter)
    return torch.optim.AdamW([
        {'params': params, 'lr': config['encoder_lr'] if encoder else config['new_lr'],
         'weight_decay': config['weight_decay'] if decay else 0.}
        for (encoder, decay), params in groups.items()])


def evaluate_streams(model, dataset, streams, inference, device, predictions_dir=None,
                     provide_document_metadata=False, collect_embeddings=False, collect_window_stats=False):
    evaluator = Evaluator(inference.start_threshold)
    if predictions_dir is not None:
        Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for i, stream in enumerate(streams):
        metadata = {}
        if provide_document_metadata:
            from .data import page_edge_targets
            targets = page_edge_targets(dataset.index, stream['rows'])
            metadata = dict(document_starts=targets['page_start'], document_ends=targets['page_end'])
        result = predict_stream(model, dataset, stream['rows'], inference, device, **metadata,
                                collect_embeddings=collect_embeddings, collect_window_stats=collect_window_stats)
        documents = {(d.library, d.doc): d for d in dataset.index.documents_for_rows(stream['rows'])}
        evaluator.add(stream['id'], dataset.index.pages(stream['rows']), result, documents,
                      stream['censored_left'], stream['censored_right'])
        if predictions_dir is not None:
            # Filenames use ordinal positions, not untrusted manifest IDs or source paths.
            np.savez_compressed(Path(predictions_dir) / f'{i:06d}.npz',
                                stream_id=stream['id'], rows=np.asarray(stream['rows'], np.int64), **result)
        del result, documents
    report = evaluator.report()
    report['seconds'] = time.perf_counter() - start
    report['unique_pages_per_second'] = sum(s['pages'] for s in report['streams']) / report['seconds']
    return report


def provenance(index, manifest):
    import hashlib
    source = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')) + sorted(
            (Path(__file__).parents[1] / 'training_store').glob('*.py')) + [
            Path(__file__).parents[1] / 'embeddings/text/page_stats/page_stats.py']:
        source.update(p.name.encode())
        source.update(p.read_bytes())
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    return {'index_sha256': index.fingerprint(), 'manifest_sha256': file_sha256(manifest),
            'code_sha256': source.hexdigest(), 'git_revision': rev.stdout.strip(),
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy')}}


def train(config, resume=None, stop_after=None):
    weights_by_stage = {stage: stage_loss_weights(config, stage) for stage in ('pretrain', 'finetune')}
    parent_size = config.get('sampling_window_size', config['window_size'])
    chunked = 'sampling_window_size' in config
    if chunked and (not 1 <= config['window_size'] <= parent_size or config.get('scan_borders')
                    or config.get('context_microbatch_size', 0) < 1):
        raise ValueError('invalid fixed-parent context experiment')
    if chunked and any(w.boundary_consistency for w in weights_by_stage.values()):
        raise ValueError('chunked training does not support boundary consistency')
    if config['window_size'] == 1 and (not chunked or any(w.starts for w in weights_by_stage.values())):
        raise ValueError('W1 requires fixed-parent chunks and disabled pairwise boundary loss')
    if any(w.boundary_consistency for w in weights_by_stage.values()) and not config.get('auxiliary_edges', False):
        raise ValueError('boundary consistency requires auxiliary_edges configuration')
    dropout = config.get('pagination_dropout', {})
    # Validate before opening large stores or creating a tracking task.
    pagination_dropout({'pagination': torch.zeros(0, 0, 5)}, **dropout)
    modality_config = config.get('modality_dropout', {})
    modality_dropout({'presence': torch.zeros(0, 0, 3)}, **modality_config)
    scattered_config = config.get('scattered_dropout', {})
    scattered_dropout({name: torch.zeros(0, 0, 3) for name in ('presence', 'image', 'text', 'stats')},
                      **scattered_config)
    if any(dropout.values()) and not config.get('pagination'):
        raise ValueError('pagination dropout requires pagination configuration')
    precision = config.get('precision', 'float32')
    if precision not in ('float32', 'float16'):
        raise ValueError('supported precision: float32 or float16')
    if precision == 'float16' and not config['device'].startswith('cuda'):
        raise ValueError('float16 training requires CUDA')
    if (config['pretrain_steps'] < 0 or config['finetune_steps'] < 0
            or config['pretrain_steps'] + config['finetune_steps'] < 1
            or config['batch_size'] < 1 or config['validate_every'] < 1
            or (stop_after is not None and stop_after < 1)):
        raise ValueError('positive training budget/batch/validation interval required')
    torch.set_num_threads(config.get('cpu_threads', 2))
    seed_all(config['seed'])
    device = config['device']
    torch.use_deterministic_algorithms(config.get('deterministic', False))
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    dataset = make_dataset(index, features, config)
    streams = load_streams(config['dev_manifest'], index, 'dev')
    inference = InferenceConfig(**config['inference'])
    from .scan_sampling import ScanBatchSampler
    complete_scans = config.get('scan_borders') or config.get('boundary_features')
    sampler_class = ScanBatchSampler if complete_scans else WindowSampler
    scan_settings = dict(config.get('scan_batches', {}))
    if complete_scans:
        if ('border_tokens' in scan_settings
                and scan_settings['border_tokens'] != bool(config.get('scan_borders'))):
            raise ValueError('scan tiling representation differs from model configuration')
        scan_settings['border_tokens'] = bool(config.get('scan_borders'))
    if scan_settings.get('metadata_probability', 0) and not config.get('boundary_features'):
        raise ValueError('document metadata requires five-bit boundary features')
    sampler = sampler_class(index, parent_size, config['seed'],
                            variable_lengths=config.get('variable_lengths', False),
                            edge_crops=config.get('edge_crops'),
                            **({'scan_batches': scan_settings} if complete_scans else {}))
    context_config = config.get('context_evaluation', {})
    diagnostic_blocks = (context_blocks(index, streams, context_config.get('sample_pages', 2048))
                         if context_config.get('enabled', False) else [])
    prov = provenance(index, config['dev_manifest'])
    prov['vector_sources'] = features.source_provenance()
    if config.get('training_store'):
        if index.fingerprint() != dataset.plan['index_sha256']:
            raise ValueError('training catalogue checksum mismatch')
        prov['training_store'] = dataset.source_provenance()
    if config.get('pagination'):
        prov['pagination'] = dataset.pagination_provenance()
    run_dir = Path(config['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        model, state = load_checkpoint(resume, device)
        if state['config'] != config or state['provenance'] != prov or state['features'] != features.state_dict():
            raise ValueError('resume config/data/code/environment/feature schema differs from checkpoint')
        step, best = state['step'], state['best']
    else:
        if (run_dir / 'latest.pt').exists() or (run_dir / 'metrics.jsonl').exists():
            raise FileExistsError('run already exists; resume explicitly or choose a new run_dir')
        backbone_config = dict(config['backbone'])
        kind = backbone_config.pop('kind', 'bert')
        backbone = (BertBackbone.create(**backbone_config) if kind == 'bert' else
                    PageBackbone(**backbone_config) if kind == 'page' else None)
        if backbone is None:
            raise ValueError('unknown backbone kind')
        model = PageTagger(backbone, getattr(dataset, 'model_dims', features.dims), config.get('fusion', 'linear'),
                           auxiliary_edges=config.get('auxiliary_edges', False),
                           scan_borders=config.get('scan_borders', False),
                           enabled_modalities=config.get('enabled_modalities')).to(device)
        step, best = 0, -1.
    model.compute_precision = precision
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'float16', init_scale=1024.)
    model.grad_scaler = scaler
    if resume and state.get('grad_scaler'):
        scaler.load_state_dict(state['grad_scaler'])
    optimizer = make_optimizer(model, config['optimizer'])
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    saved_training = state.get('training') if resume else None
    checkpoint_rng = state['rng'] if resume else None
    if resume:
        optimizer.load_state_dict(state['optimizer'])
        sampler.load_state_dict(state['sampler'])
        del state  # do not retain legacy diagnostic histories or CPU optimizer tensors
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (run_dir / 'provenance.json').write_text(json.dumps(prov | {'model': model.specification(),
        'features': features.state_dict(), 'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}, indent=2) + '\n')
    tracking = Tracking(config, prov, bool(resume))
    # ClearML Task.init calls make_deterministic, even when framework auto-connect
    # is disabled. Restore after SDK initialization so resumed dropout matches.
    if resume:
        restore_rng(checkpoint_rng)
        del checkpoint_rng
    total = config['pretrain_steps'] + config['finetune_steps']
    limit = min(total, stop_after) if stop_after is not None else total
    warmup = max(1, config['optimizer'].get('warmup_steps',
                     math.ceil(total * config['optimizer']['warmup_fraction'])))
    progress = saved_training or {
        'stage': 'pretrain' if config['pretrain_steps'] else 'finetune',
        'stage_step': 0, 'stage_best': -1., 'bad_validations': 0, 'finished': False}
    if resume and saved_training is None:
        progress['stage'] = 'pretrain' if step < config['pretrain_steps'] else 'finetune'
        progress['stage_step'] = step if progress['stage'] == 'pretrain' else step - config['pretrain_steps']
    if chunked and not resume:
        # A memory interruption before the first validation must still be resumable.
        save_checkpoint(run_dir / 'initial.pt', model, optimizer, sampler, step, best,
                        config, prov, features.state_dict(), progress)
        save_checkpoint(run_dir / 'latest.pt', model, optimizer, sampler, step, best,
                        config, prov, features.state_dict(), progress)
    loader, batches = training_loader(dataset, sampler, config)
    model.train()

    def finish_phase():
        # Preserve both the actual last update and the selected next-phase state.
        # phase_end makes a completed-pretraining snapshot directly resumable even
        # if the process stops before publishing the fine-tuning initialization.
        nonlocal progress
        stage = progress['stage']
        if stage != 'pretrain' or not config['finetune_steps']:
            progress['finished'] = True
        final_path = run_dir / (stage + '_final.pt')
        if not final_path.exists():
            save_checkpoint(final_path, model, optimizer, sampler, step, best,
                            config, prov, features.state_dict(), progress)
        if stage == 'pretrain' and config['finetune_steps']:
            if not (run_dir / 'pretrain_best.pt').exists():
                raise RuntimeError('pretraining budget ended before any eligible validation; increase its budget')
            # Preserve the existing selection and RNG-consumption behavior.
            restored_model, selected = load_checkpoint(run_dir / 'pretrain_best.pt', 'cpu')
            model.load_state_dict(selected['model'])
            optimizer.load_state_dict(selected['optimizer'])
            if selected.get('grad_scaler'):
                scaler.load_state_dict(selected['grad_scaler'])
            del restored_model, selected
            progress = {'stage': 'finetune', 'stage_step': 0, 'stage_best': -1.,
                        'bad_validations': 0, 'finished': False}
            sampler.set_stage('finetune')
            initial_path = run_dir / 'finetune_init.pt'
            if not initial_path.exists():
                save_checkpoint(initial_path, model, optimizer, sampler, step, best,
                                config, prov, features.state_dict(), progress)

    try:
        if progress.get('phase_end'):
            finish_phase()
            save_checkpoint(run_dir / 'latest.pt', model, optimizer, sampler, step, best,
                            config, prov, features.state_dict(), progress)
        while (step < limit or progress.get('pending_validation')) and not progress['finished']:
            stage = progress['stage']
            loss_weights = weights_by_stage[stage]
            if sampler.stage != stage:
                sampler.set_stage(stage)
            stage_budget = config[stage + '_steps']
            interval = config.get(stage + '_validate_every', config['validate_every'])
            pending = progress.get('pending_validation')
            count = 0 if pending else min(interval - progress['stage_step'] % interval,
                        stage_budget - progress['stage_step'], limit - step)
            batches.batches = count
            # All prefetched draws in this finite segment are consumed before checkpoints,
            # evaluation, early stopping or stage changes. No sampler snapshot per batch.
            if pending:
                record = pending
            else:
                record = None
                iterator = iter(loader)
                for _ in range(count):
                    tick = time.perf_counter()
                    raw_batch = next(iterator)
                    batch = to_device(raw_batch, device,
                                      feature_dtype=torch.float32 if precision == 'float32' else None)
                    io_seconds = time.perf_counter() - tick
                    for group, lr in zip(optimizer.param_groups, base_lrs):
                        group['lr'] = lr * min((step + 1) / warmup, 1.)
                    optimizer.zero_grad(set_to_none=True)
                    augmented = pagination_dropout(batch['features'], **dropout)
                    augmented = modality_dropout(augmented, **modality_config)
                    augmented = scattered_dropout(augmented, **scattered_config)
                    if chunked:
                        output = None
                        losses, denominators = backward_chunks(
                            model, batch, augmented, config['window_size'],
                            config['context_microbatch_size'], loss_weights, scaler, device)
                    else:
                        with autocast(model, device):
                            output = model(augmented, batch['attention_mask'])
                        losses, denominators = task_losses(output, batch, loss_weights)
                        if not torch.isfinite(losses['total']):
                            raise FloatingPointError('nonfinite training loss')
                        scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad'],
                                                          error_if_nonfinite=precision == 'float32')
                    scaler.step(optimizer)
                    scaler.update()
                    if not torch.isfinite(norm):
                        progress['overflow_skips'] = progress.get('overflow_skips', 0) + 1
                        print(json.dumps({'overflow_skipped': progress['overflow_skips'],
                                          'step': step, 'scale': scaler.get_scale()}), flush=True)
                        if scaler.get_scale() < 1 or progress['overflow_skips'] > 20:
                            raise FloatingPointError('repeated FP16 gradient overflow')
                        del raw_batch, batch, augmented, output, losses, denominators, norm
                        continue
                    step += 1
                    progress['stage_step'] += 1
                    loss_values = {k: float(v.detach()) for k, v in losses.items()}
                    elapsed = time.perf_counter() - tick
                    record = {'step': step, 'stage': stage,
                              'stage_step': progress['stage_step'],
                              'loss': loss_values,
                              'loss_weights': asdict(loss_weights),
                              'valid_weight': {k: float(v) for k, v in denominators.items()},
                              'gradient_norm': float(norm), 'io_seconds': io_seconds,
                              'storage_pages': raw_batch.get('storage_pages', {}),
                              'loss_scale': scaler.get_scale(),
                              'pages_per_second': int(raw_batch['attention_mask'].sum()) / elapsed,
                              'sampler': dict(sampler.counters), 'coverage_epochs': sampler.epoch}
                    if chunked:
                        record['context_partition'] = dict(parent_size=parent_size,
                            context_size=config['window_size'], microbatch=config['context_microbatch_size'])
                    # The device copy and scalar logging are complete. Do not retain
                    # the consumer's pinned batch (or last training graph) in validation.
                    del raw_batch, batch, augmented, output, losses, denominators, norm
                    with (run_dir / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record, allow_nan=False) + '\n')
                    if step % config.get('log_every', 25) == 0 or step == 1:
                        print(json.dumps({k: record[k] for k in
                            ('step', 'stage', 'loss', 'pages_per_second', 'coverage_epochs')}), flush=True)
                        tracking.report(record)
                # Force iterator exhaustion even with persistent workers, then checkpoint safely.
                try:
                    next(iterator)
                    raise RuntimeError('loader exceeded segment budget')
                except StopIteration:
                    pass
                if record is None or (progress['stage_step'] % interval and step < limit
                                      and progress['stage_step'] < stage_budget):
                    # Overflowed draws count as exposure, never as optimizer updates.
                    # Finish the requested update segment before validation/checkpointing.
                    continue
                # Preserve model/optimizer/sampler before potentially failing evaluation.
                # Resume evaluates this pending boundary before drawing any more windows.
                progress['pending_validation'] = record
                save_checkpoint(run_dir / 'latest.pt', model, optimizer, sampler, step, best,
                                config, prov, features.state_dict(), progress)
            host_cache = release_host_cache(device)
            if host_cache:
                print(json.dumps({'step': step, 'validation_host_cache': host_cache}), flush=True)
            report = evaluate_streams(model, dataset, streams, inference, device)
            if config.get('document_metadata_evaluation'):
                report['metadata_provided'] = evaluate_streams(
                    model, dataset, streams, inference, device, provide_document_metadata=True)
            if diagnostic_blocks:
                lengths = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                           if n <= config['window_size']]
                # Rotate the order if a slow machine exhausts the wall-time budget.
                offset = (step // interval) % len(lengths)
                lengths = lengths[offset:] + lengths[:offset]
                report['contexts'] = evaluate_contexts(
                    model, dataset, diagnostic_blocks, lengths, device,
                    batch_size=config['inference']['batch_size'],
                    seconds_budget=report['seconds'] * context_config.get('budget_fraction', .1))
            score = report['curated_type']['weighted_f1']
            if score is None:
                raise ValueError('development streams contain no curated type targets')
            progress.pop('pending_validation', None)
            policy = config.get('stopping', {}).get(stage, {})
            eligible = sampler.epoch >= policy.get('min_coverage_epochs', 0)
            improved = score > best
            stage_improved = eligible and score > progress['stage_best']
            best = max(best, score)
            if eligible:
                progress['stage_best'] = max(progress['stage_best'], score)
            progress['bad_validations'] = (0 if stage_improved or not eligible else
                                           progress['bad_validations'] + 1)
            record['dev'] = report
            with (run_dir / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps({'step': step, 'stage': stage,
                              'dev_curated_weighted_f1': score,
                              'dev_starts_f1': report.get('starts', {}).get('all', {}).get('f1'),
                              'best': best}), flush=True)
            tracking.report(record)
            args = (model, optimizer, sampler, step, best, config, prov, features.state_dict(), progress)
            if stage_improved:
                save_checkpoint(run_dir / (stage + '_best.pt'), *args)
            if improved:
                save_checkpoint(run_dir / 'best.pt', *args)
            plateau = ('patience' in policy and eligible and
                       progress['bad_validations'] >= policy['patience'])
            ended = progress['stage_step'] >= stage_budget or plateau
            if ended:
                reason = 'validation_plateau' if plateau else 'stage_budget'
                print(json.dumps({'stage_finished': stage, 'step': step, 'reason': reason}), flush=True)
                progress['phase_end'] = reason
                finish_phase()
            save_checkpoint(run_dir / 'latest.pt', model, optimizer, sampler, step, best,
                            config, prov, features.state_dict(), progress)
        if progress['finished'] and diagnostic_blocks:
            selected_path = run_dir / ('finetune_best.pt' if config['finetune_steps'] else 'pretrain_best.pt')
            selected = torch.load(selected_path, map_location='cpu', weights_only=False)
            model.load_state_dict(selected['model'])
            selected_step = selected['step']
            del selected
            if config.get('document_metadata_evaluation'):
                from .compare_page_context import compare
                metadata_dir = run_dir / f'final-provided-context-step{selected_step}'
                if not (metadata_dir / 'report.json').exists():
                    if metadata_dir.exists():
                        metadata_dir.rename(run_dir / (metadata_dir.name + f'-incomplete-{time.time_ns()}'))
                    metadata_report = compare(model, dataset, streams, inference, metadata_dir,
                                              device, provide_document_metadata=True)
                    metadata_report['checkpoint_step'] = selected_step
                    (metadata_dir / 'report.json').write_text(json.dumps(metadata_report, indent=2) + '\n')
            if config.get('final_scan_context_comparison'):
                from .compare_page_context import compare
                comparison_dir = run_dir / f'final-scan-context-step{selected_step}'
                # Final reports may be revisited after a completed-run resume.
                comparison_report = comparison_dir / 'report.json'
                if not comparison_report.exists():
                    if comparison_dir.exists():
                        # Preserve an interrupted diagnostic; never let its partial
                        # arrays prevent resuming the completed training checkpoint.
                        comparison_dir.rename(run_dir / (comparison_dir.name + f'-incomplete-{time.time_ns()}'))
                    comparison = compare(model, dataset, streams, inference, comparison_dir, device)
                    comparison['checkpoint_step'] = selected_step
                    comparison['manifest_sha256'] = prov['manifest_sha256']
                    comparison['index_sha256'] = prov['index_sha256']
                    comparison_report.write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
                else:
                    comparison = json.loads(comparison_report.read_text())
                if tracking.task is not None:
                    for mode, scores in comparison['modes'].items():
                        for metric in ('weighted_f1', 'accuracy', 'macro_f1'):
                            tracking.logger.report_scalar('final_scan_context/' + mode, metric,
                                                          scores['curated_type'][metric], selected_step)
            final_report = evaluate_contexts(
                model, dataset, context_blocks(index, streams, context_config.get('final_sample_pages', 8192)),
                [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if n <= config['window_size']],
                device, batch_size=config['inference']['batch_size'])
            final_report['checkpoint_step'] = selected_step
            (run_dir / 'final-context-evaluation.json').write_text(json.dumps(final_report, allow_nan=False) + '\n')
            tracking.report_contexts(final_report, selected_step, prefix='final_context')
            if config.get('auxiliary_edges', False):
                from .context_evaluation import document_edge_blocks
                edge_report = evaluate_contexts(model, dataset,
                    document_edge_blocks(index, config['window_size'], context_config.get('final_sample_pages', 8192)),
                    [config['window_size']], device, batch_size=config['inference']['batch_size'])
                edge_report.update(checkpoint_step=selected_step,
                                   description='dev document prefixes/tails and interior negatives; no neighbouring document visible')
                (run_dir / 'final-document-edges.json').write_text(json.dumps(edge_report, allow_nan=False) + '\n')
                tracking.report_contexts(edge_report, selected_step, prefix='final_document_edges')
            if config.get('pagination') and config.get('pagination_robustness_evaluation', False):
                missing_report = evaluate_contexts(
                    model, dataset, context_blocks(index, streams, context_config.get('final_sample_pages', 8192)),
                    [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if n <= config['window_size']],
                    device, batch_size=config['inference']['batch_size'], omit_pagination=True)
                missing_report['checkpoint_step'] = selected_step
                (run_dir / 'final-context-without-pagination.json').write_text(
                    json.dumps(missing_report, allow_nan=False) + '\n')
                tracking.report_contexts(missing_report, selected_step, prefix='final_context_without_pagination')
            if config.get('robustness_evaluation', False):
                from .robustness import evaluate_robustness
                evaluate_robustness(model, dataset,
                    context_blocks(index, streams, context_config.get('final_sample_pages', 8192)),
                    config['window_size'], run_dir, selected_step, device,
                    config['inference']['batch_size'], tracking)
    finally:
        if getattr(loader, '_iterator', None) is not None:
            loader._iterator._shutdown_workers()
        tracking.close(failed=sys.exc_info()[0] is not None)
    return run_dir / 'latest.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--config', required=True)
    tr.add_argument('--resume')
    tr.add_argument('--stop-after', type=int, help='Stop at this absolute update, saving resumable state')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--manifest', required=True)
    ev.add_argument('--split', choices=['dev', 'test'], required=True)
    ev.add_argument('--output', required=True)
    ev.add_argument('--predictions-dir')
    ev.add_argument('--training-store', help='Relocated, completed native store; identity and catalogue are verified')
    ev.add_argument('--inference-config', help='Optional JSON InferenceConfig for development merge comparisons')
    ev.add_argument('--device', default='cpu')
    ev.add_argument('--collect-embeddings', action='store_true')
    ev.add_argument('--collect-window-stats', action='store_true')
    pred = sub.add_parser('predict')
    pred.add_argument('--checkpoint', required=True)
    pred.add_argument('--pages', required=True, help='Scan-ordered JSONL: library, page, path')
    pred.add_argument('--output', required=True, help='Output .npz with probabilities/labels/segments')
    pred.add_argument('--sources', help='Optional JSON mapping image/text/stats to new vector-store specs')
    pred.add_argument('--device', default='cpu')
    pred.add_argument('--collect-embeddings', action='store_true')
    pred.add_argument('--collect-window-stats', action='store_true')
    args = parser.parse_args()
    if args.command == 'train':
        train(json.loads(Path(args.config).read_text()), args.resume, args.stop_after)
    elif args.command == 'evaluate':
        model, state = load_checkpoint(args.checkpoint, args.device)
        config, dataset = evaluation_dataset(state, args.training_store)
        settings = json.loads(Path(args.inference_config).read_text()) if args.inference_config else config['inference']
        inference = InferenceConfig(**settings)
        streams = load_streams(args.manifest, dataset.index, args.split)
        report = evaluate_streams(model, dataset, streams, inference,
                                 args.device, args.predictions_dir,
                                 collect_embeddings=args.collect_embeddings,
                                 collect_window_stats=args.collect_window_stats)
        report['inference'] = asdict(inference)
        report['manifest_sha256'] = file_sha256(args.manifest)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    else:
        model, state = load_checkpoint(args.checkpoint, args.device)
        settings = dict(state['config']['features'])
        if args.sources:
            settings['sources'] = json.loads(Path(args.sources).read_text())
        features = FeaturePipeline.from_config(settings)
        if features.state_dict() != state['features']:
            raise ValueError('checkpoint/feature schema mismatch')
        pages = []
        with open(args.pages) as f:
            for i, line in enumerate(f):
                p = json.loads(line)
                pages.append(Page(p['library'], '', p['page'], i, path=p.get('path', ''), pagenum=p.get('pagenum', '')))
        result = tag_pages(model, features, pages, InferenceConfig(**state['config']['inference']), args.device,
                           pagination_cap=state['config'].get('pagination', {}).get('cap', 10000),
                           collect_embeddings=args.collect_embeddings,
                           collect_window_stats=args.collect_window_stats)
        np.savez_compressed(args.output, page_types=np.array(PAGE_TYPES), placements=np.array(PLACEMENTS), **result)


if __name__ == '__main__':
    main()
