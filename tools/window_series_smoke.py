"""Real-store preflight for one r22 window-series run: six CUDA updates over both phases and all final evaluations.

Audits the training forwards (window length, physical positions, all three signals), the cutting of the sampled
512-page windows (coverage scans with overlap, anchors consecutive, every page weighted once per update), finite
losses with an active boundary loss, the unchanged 512 sampler with the regular tail, the deployed evaluation
policy, the phase checkpoints and the initial parameters.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

import page_sequence.context_training as context_training
from page_sequence.inference import deployment_inference
from page_sequence.models import PageTagger
from page_sequence.runner import train


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'smoke-config.json').read_text())
    window = config['window_size']
    chunked = True
    assert config['sampling_window_size'] == 512 and config['context_coverage_stride'] == window // 2
    run = Path(config['run_dir'])
    assert config['inference'] == asdict(deployment_inference(window, config['inference']['batch_size'],
                                                              config['inference']['start_threshold']))
    assert config['scan_batches']['tail'] == 'regular'
    audit = {'train_forwards': 0, 'eval_forwards': 0, 'max_physical_positions': 0, 'updates_cut': 0,
             'pieces': 0, 'parents': 0, 'coverage_parents': 0}
    original_forward = PageTagger.forward

    def checked_forward(self, features, mask, **kwargs):
        assert self.enabled_modalities == ('image', 'text', 'stats')
        assert mask.shape[1] <= window
        key = 'train_forwards' if self.training else 'eval_forwards'
        audit[key] += 1
        if self.training:
            if chunked:
                assert mask.shape[1] == window
            audit['max_physical_positions'] = max(audit['max_physical_positions'], mask.numel())
            if 'initial_parameter_sha256' not in audit:
                digest = hashlib.sha256()
                for name, tensor in self.state_dict().items():
                    digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
                audit['initial_parameter_sha256'] = digest.hexdigest()
        return original_forward(self, features, mask, **kwargs)

    original_cut = context_training.scan_context

    def checked_cut(batch, size, parent_stride, coverage_stride):
        pieces = original_cut(batch, size, parent_stride, coverage_stride)
        assert (size, parent_stride, coverage_stride) == (window, 256, window // 2)
        totals = Counter()
        for rows, weights in zip(pieces['rows'], pieces['weight'].tolist()):
            for row, weight in zip(rows, weights):
                totals[row] += weight
        assert all(abs(total - 1) < 1e-5 for total in totals.values())
        assert set(totals) == {row for rows in batch['rows'] for row in rows}
        assert pieces['attention_mask'].sum() >= batch['attention_mask'].sum()
        audit['updates_cut'] += 1
        audit['pieces'] += len(pieces['rows'])
        audit['coverage_parents'] += sum(source == 'coverage' for source in batch['sources'])
        audit['parents'] += len(batch['sources'])
        return pieces

    PageTagger.forward = checked_forward
    context_training.scan_context = checked_cut
    train(config)
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1, 7))
    for record in records:
        assert all(math.isfinite(v) for v in record['loss'].values())
        assert math.isfinite(record['gradient_norm'])
        assert record['loss_weights']['starts'] > 0 and record['valid_weight']['starts'] > 0
        assert record['storage_pages']['fallback'] == 0 and record['storage_pages']['nvme'] > 0
        assert record['loss_weights']['weak_type'] == (.2 if record['stage'] == 'pretrain' else .05)
        if chunked:
            assert record['context_partition'] == dict(parent_size=512, context_size=window, microbatch=32768,
                                                       parent_stride=256, coverage_stride=window // 2)
        else:
            assert 'context_partition' not in record
    assert audit['updates_cut'] == (len(records) if chunked else 0)
    if chunked:
        assert audit['coverage_parents'] > 0 and audit['pieces'] > audit['parents']
    assert audit['max_physical_positions'] <= 32768
    for name in ('initial.pt', 'pretrain_final.pt', 'finetune_init.pt', 'finetune_final.pt', 'latest.pt'):
        if name == 'initial.pt' and not chunked:
            continue
        state = torch.load(run / name, map_location='cpu', weights_only=False)
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        assert state['sampler']['window_size'] == 512
        assert state['sampler']['scan_batches']['tail'] == 'regular'
        assert state['specification']['dims']['stats'] == 180
    assert state['training']['finished']
    for line in (run / 'validation.jsonl').read_text().splitlines():
        dev = json.loads(line)['dev']
        assert dev['starts']['all']['n'] > 0 and dev['curated_type']['n'] > 0
    assert list(run.glob('final-scan-context-step*/report.json'))
    assert list(run.glob('final-provided-context-step*/report.json'))
    assert (run / 'final-context-evaluation.json').exists()
    expected = json.loads((root / 'expected-initialization.json').read_text())
    assert audit['initial_parameter_sha256'] == expected['sha256']
    audit.update(peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                 peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                 allocator_config=os.environ.get('PYTORCH_ALLOC_CONF'))
    (root / 'smoke-passed.json').write_text(json.dumps(dict(
        passed=True, context_microbatch_size=config.get('context_microbatch_size'), **audit), indent=2) + '\n')


if __name__ == '__main__':
    main()
