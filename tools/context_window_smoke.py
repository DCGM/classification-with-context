"""Real-store preflight: independent pages, inactive head, preserved parent batches."""
import hashlib
import json
import math
from pathlib import Path

import torch

from page_sequence.models import PageTagger
from page_sequence.runner import train


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'smoke-config.json').read_text())
    run = Path(config['run_dir'])
    audit = {'train_forwards': 0, 'eval_forwards': 0, 'max_physical_batch': 0}
    original = PageTagger.forward
    initial_head = None

    def checked(self, features, mask, **kwargs):
        nonlocal initial_head
        assert mask.shape[1] == 1
        assert self.enabled_modalities == tuple(config['enabled_modalities'])
        assert mask.all()
        for i, name in enumerate(('image', 'text', 'stats')):
            if name not in self.enabled_modalities:
                assert not features[name].any() and not features['presence'][..., i].any()
        key = 'train_forwards' if self.training else 'eval_forwards'
        audit[key] += 1
        if self.training:
            audit['max_physical_batch'] = max(audit['max_physical_batch'], mask.shape[0])
            if initial_head is None:
                initial_head = {k: v.detach().cpu().clone() for k, v in self.start_head.state_dict().items()}
                digest = hashlib.sha256()
                for name, tensor in self.state_dict().items():
                    digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
                audit['initial_parameter_sha256'] = digest.hexdigest()
        return original(self, features, mask, **kwargs)

    PageTagger.forward = checked
    train(config)
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1, 7))
    for record in records:
        assert all(math.isfinite(v) for v in record['loss'].values())
        assert math.isfinite(record['gradient_norm'])
        assert record['loss']['starts'] == record['valid_weight']['starts'] == 0
        assert record['storage_pages']['fallback'] == 0
        assert record['storage_pages']['nvme'] > 0
        assert record['context_partition']['parent_size'] == 512
        assert record['loss_weights']['weak_type'] == (.2 if record['stage'] == 'pretrain' else .05)
    for name in ('pretrain_final.pt', 'finetune_init.pt', 'finetune_final.pt', 'latest.pt'):
        state = torch.load(run / name, map_location='cpu', weights_only=False)
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        assert state['sampler']['window_size'] == 512
        assert state['specification']['dims']['stats'] == 180
        assert len(state['features']['stats_extra_groups']) == 11
        for key, tensor in initial_head.items():
            assert torch.equal(tensor, state['model']['start_head.' + key])
    assert state['training']['finished']
    for line in (run / 'validation.jsonl').read_text().splitlines():
        dev = json.loads(line)['dev']
        assert dev['starts']['all']['n'] == 0
        assert dev['metadata_provided']['starts']['all']['n'] == 0
        assert dev['exact_stream_fraction'] is None
        assert dev['curated_type']['n'] > 0
    assert audit['train_forwards'] >= len(records)
    if config['context_microbatch_size'] >= config['batch_size'] * config['sampling_window_size']:
        assert audit['train_forwards'] == len(records)
    assert audit['max_physical_batch'] <= config['context_microbatch_size']
    expected = json.loads((root / 'expected-initialization.json').read_text())
    assert audit['initial_parameter_sha256'] == expected['sha256']
    (root / 'smoke-passed.json').write_text(json.dumps(dict(passed=True,
        context_microbatch_size=config['context_microbatch_size'], **audit), indent=2) + '\n')


if __name__ == '__main__':
    main()
