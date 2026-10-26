import json

import numpy as np
import pytest
import torch

from page_sequence.context_evaluation import context_blocks, evaluate_contexts
from page_sequence.data import WindowDataset
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.manifests import load_streams
from page_sequence.models import BertBackbone, PageTagger
from page_sequence.runner import train
from page_sequence.sampling import WindowSampler
from page_sequence.smoke import prepare_fixture


@pytest.mark.parametrize('maximum', [128, 256])
def test_length_distribution_and_resumption(tmp_path, maximum):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    sampler = WindowSampler(index, maximum, seed=37, variable_lengths=True)
    lengths = np.array([sampler._draw_length() for _ in range(20000)])
    for mask, expected in ((lengths == maximum, .6), ((lengths >= 32) & (lengths < maximum), .2),
                           ((lengths >= 8) & (lengths <= 31), .1),
                           ((lengths >= 2) & (lengths <= 7), .05), (lengths == 1, .05)):
        assert abs(mask.mean() - expected) < .015
    for stage in ('pretrain', 'finetune'):
        sampler.set_stage(stage)
        for _ in range(50):
            window = sampler.draw()
            assert 1 <= len(window.rows) <= sampler.draw_size <= maximum
        resumed = WindowSampler(index, maximum, variable_lengths=True)
        resumed.load_state_dict(sampler.state_dict())
        assert [sampler.draw() for _ in range(30)] == [resumed.draw() for _ in range(30)]
    with pytest.raises(ValueError, match='length policy'):
        WindowSampler(index, maximum).load_state_dict(sampler.state_dict())


def test_short_anchors_and_coverage_are_retained(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    sampler = WindowSampler(index, 128, variable_lengths=True)
    doc = sampler.docs[0]
    for n in (1, 2, 4, 8, 32, 128):
        sampler.draw_size = n
        for boundary in (False, True):
            anchor = doc.first if boundary else doc.first + 5
            rows = sampler._around(doc, anchor, boundary)
            assert anchor in rows and len(rows) <= n
            if boundary and n > 1:
                assert rows.index(anchor) > 0
    seen = set()
    while sampler.epoch == 0:
        sampler.draw_size = sampler._draw_length()
        seen.update(sampler._coverage())
    assert seen == {r for d in sampler.docs for r in range(d.first, d.first + d.length)}


def test_diagnostic_same_pages_boundary_masks_budget_and_rng(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    dataset = WindowDataset(index, FeaturePipeline.from_config(config['features']))
    streams = load_streams(config['dev_manifest'], index, 'dev')
    blocks = context_blocks(index, streams, sample_pages=16, block_size=8)
    assert blocks == context_blocks(index, streams, sample_pages=16, block_size=8)
    model = PageTagger(BertBackbone.create(**{k: v for k, v in config['backbone'].items() if k != 'kind'}),
                       dataset.features.dims)
    model.train()
    before = torch.get_rng_state().clone()
    class CountReads:
        def __init__(self):
            self.calls = 0
        def __getitem__(self, window):
            self.calls += 1
            return dataset[window]
    cached = CountReads()
    report = evaluate_contexts(model, cached, blocks, [1, 2, 4, 8], batch_size=3)
    assert cached.calls == len(blocks)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    assert report['complete']
    for n, scores in report['lengths'].items():
        assert scores['pages'] == 16 and scores['complete']
        if n == '1':
            assert scores['starts'] is None
        else:
            assert scores['starts']['n'] == 16 - scores['scans']
    empty = evaluate_contexts(model, dataset, blocks, [1, 2], seconds_budget=0)
    assert not empty['complete'] and empty['lengths']['1']['pages'] == 0
    with pytest.raises(ValueError, match='development'):
        context_blocks(index, [dict(streams[0], split='test')])
    json.dumps(report, allow_nan=False)


def test_training_records_periodic_and_selected_final_diagnostics(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    config.update(variable_lengths=True, context_evaluation={
        'enabled': True, 'sample_pages': 16, 'final_sample_pages': 32, 'budget_fraction': .1})
    train(config)
    validation = [json.loads(line) for line in (tmp_path / 'run/validation.jsonl').read_text().splitlines()]
    assert all('contexts' in record['dev'] for record in validation)
    final = json.loads((tmp_path / 'run/final-context-evaluation.json').read_text())
    selected = torch.load(tmp_path / 'run/finetune_best.pt', weights_only=False)
    assert final['complete'] and final['checkpoint_step'] == selected['step']
