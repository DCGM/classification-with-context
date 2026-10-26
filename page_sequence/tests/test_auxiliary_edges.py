import copy
import json
from pathlib import Path

import pytest
import torch

from page_sequence.context_evaluation import document_edge_blocks, evaluate_contexts
from page_sequence.data import WindowDataset, collate_windows
from page_sequence.export import export_checkpoint
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.losses import LossWeights, task_losses
from page_sequence.models import PageBackbone, PageTagger, from_specification
from page_sequence.runner import load_checkpoint, train
from page_sequence.sampling import WindowSampler
from page_sequence.schema import IGNORE, PAGE_TYPES, PLACEMENTS, Window
from page_sequence.smoke import prepare_fixture


def fixture(tmp_path):
    config = prepare_fixture(tmp_path, steps=8)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    return config, index, features, WindowDataset(index, features)


def test_targets_are_catalogue_edges_not_crop_edges_and_padding(tmp_path):
    _, index, _, dataset = fixture(tmp_path)
    doc = index.documents('fit')[0]
    head = dataset[Window(tuple(range(doc.first, doc.first + 3)))]
    tail = dataset[Window(tuple(range(doc.first + doc.length - 3, doc.first + doc.length)))]
    middle = dataset[Window((doc.first + 2,))]
    assert head['page_start'].tolist() == [1, 0, 0]
    assert head['page_end'].tolist() == [0, 0, 0]
    assert tail['page_start'].tolist() == [0, 0, 0]
    assert tail['page_end'].tolist() == [0, 0, 1]
    assert middle['page_start'].tolist() == middle['page_end'].tolist() == [0]
    batch = collate_windows([head, tail, middle])
    assert batch['page_start'][2].tolist() == [0, IGNORE, IGNORE]
    assert not batch['pair_mask'][:, 0].any()


def test_auxiliary_gradients_at_first_last_singleton_and_export(tmp_path):
    _, index, features, dataset = fixture(tmp_path)
    doc = index.documents('fit')[0]
    batch = collate_windows([dataset[Window((doc.first,))],
                             dataset[Window((doc.first + doc.length - 1,))]])
    model = PageTagger(PageBackbone(16), features.dims, auxiliary_edges=True).eval()
    output = model(batch['features'], batch['attention_mask'])
    assert output['page_start_logits'].shape == (2, 1)
    weights = LossWeights(gold_type=0, weak_type=0, placement=0, starts=0)
    losses, denominators = task_losses(output, batch, weights)
    assert denominators['starts'] == 0
    assert denominators['page_start'] == denominators['page_end'] == 2
    losses['total'].backward()
    assert model.page_start_head.weight.grad.abs().sum() > 0
    assert model.page_end_head.weight.grad.abs().sum() > 0
    assert model.backbone.layers[0].weight.grad.abs().sum() > 0
    spec = model.specification()
    restored = from_specification(spec)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.auxiliary_edges
    source, exported = tmp_path / 'training.pt', tmp_path / 'portable.pt'
    torch.save({'format': 1, 'page_types': PAGE_TYPES, 'placements': PLACEMENTS,
                'specification': spec, 'features': features.state_dict(), 'model': model.state_dict(),
                'config': {}}, source)
    export_checkpoint(source, exported)
    artifact = torch.load(exported, weights_only=True)
    assert 'auxiliary_edges' not in artifact['specification']
    legacy = from_specification(artifact['specification']).eval()
    legacy.load_state_dict(artifact['model'], strict=True)
    predicted = legacy(batch['features'], batch['attention_mask'])
    for key in ('type_logits', 'placement_logits', 'starts_document_logits', 'pair_mask'):
        torch.testing.assert_close(output[key], predicted[key], rtol=0, atol=0)


def test_edge_crops_keep_annotations_and_coverage_and_resume(tmp_path):
    _, index, _, _ = fixture(tmp_path)
    policy = {'start_probability': .5, 'end_probability': .5}
    sampler = WindowSampler(index, 12, variable_lengths=True, edge_crops=policy)
    for _ in range(500):
        sampler.draw()
    assert sampler.counters['edge_crop_applied:start'] > 0
    assert sampler.counters['edge_crop_applied:end'] > 0
    other = WindowSampler(index, 12, variable_lengths=True, edge_crops=policy)
    other.load_state_dict(sampler.state_dict())
    assert [sampler.draw() for _ in range(40)] == [other.draw() for _ in range(40)]
    with pytest.raises(ValueError, match='edge crop policy'):
        WindowSampler(index, 12, variable_lengths=True).load_state_dict(sampler.state_dict())
    # Whole corpus coverage is unchanged by the optional cropping policy.
    a, b = WindowSampler(index, 12), WindowSampler(index, 12, edge_crops=policy)
    for _ in range(30):
        assert a._coverage() == b._coverage()


@pytest.mark.parametrize('kind', ['start', 'end'])
def test_edge_crops_never_discard_an_annotated_page(tmp_path, monkeypatch, kind):
    _, index, _, _ = fixture(tmp_path)
    sampler = WindowSampler(index, 20, edge_crops={kind + '_probability': 1.})
    docs = sampler.docs[:2]
    rows = tuple(range(docs[0].first, docs[0].first + docs[0].length)) + tuple(
        range(docs[1].first, docs[1].first + docs[1].length))
    gold_rows = {row for row, page in zip(rows, index.pages(rows)) if page.annotated_type}
    monkeypatch.setattr(sampler, '_anchor', lambda source: rows)
    for _ in range(10):
        sampler.pending_sources = ['annotated']
        assert gold_rows <= set(sampler.draw().rows)


def test_targeted_edge_panel_and_auxiliary_context_metrics(tmp_path):
    _, index, features, dataset = fixture(tmp_path)
    blocks = document_edge_blocks(index, 12, sample_pages=100)
    assert blocks == document_edge_blocks(index, 12, sample_pages=100)
    for rows in blocks:
        pages = index.pages(rows)
        assert len({page.doc_key for page in pages}) == 1
    model = PageTagger(PageBackbone(16), features.dims, auxiliary_edges=True)
    report = evaluate_contexts(model, dataset, blocks, [1, 12])
    assert report['complete']
    for length, scores in report['lengths'].items():
        for task in ('page_start', 'page_end'):
            assert scores['auxiliary_edges'][task]['all']['n'] == scores['pages']
            assert scores['auxiliary_edges'][task]['scan_edge']['n'] == scores['scans']


def test_auxiliary_training_resume_and_final_diagnostics(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=8)
    config.update(auxiliary_edges=True, edge_crops={'start_probability': .1, 'end_probability': .1},
                  context_evaluation={'enabled': True, 'sample_pages': 20, 'final_sample_pages': 40})
    train(config)
    _, original = load_checkpoint(Path(config['run_dir']) / 'latest.pt')
    resumed = copy.deepcopy(config)
    resumed['run_dir'] = str(tmp_path / 'resumed')
    train(resumed, stop_after=4)
    train(resumed, resume=Path(resumed['run_dir']) / 'latest.pt')
    _, restored = load_checkpoint(Path(resumed['run_dir']) / 'latest.pt')
    for key, value in original['model'].items():
        torch.testing.assert_close(value, restored['model'][key], rtol=0, atol=0)
    report = json.loads((Path(config['run_dir']) / 'final-document-edges.json').read_text())
    assert report['complete'] and 'auxiliary_edges' in report['lengths']['12']
