import copy
import json

import numpy as np
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from page_sequence.data import WindowDataset, collate_windows
from page_sequence.evaluation import Evaluator
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex, build_index
from page_sequence.inference import InferenceConfig, StreamMerger, predict_stream, tag_pages, window_spans
from page_sequence.losses import task_losses
from page_sequence.manifests import load_streams
from page_sequence.models import BertBackbone, PageTagger
from page_sequence.runner import train, load_checkpoint
from page_sequence.sampling import WindowSampler
from page_sequence.schema import IGNORE, LABEL2ID, PAGE_TYPES, Page, Window
from page_sequence.smoke import prepare_fixture


@pytest.fixture
def fixture_data(tmp_path):
    config = prepare_fixture(tmp_path, steps=4)
    index = PageIndex(config['index'])
    features = FeaturePipeline.from_config(config['features'])
    return config, index, features, WindowDataset(index, features)


def tiny_model(dims):
    return PageTagger(BertBackbone.create(pretrained=False, config={
        'vocab_size': 8, 'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 4,
        'intermediate_size': 32, 'max_position_embeddings': 128,
        'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.}), dims)


def test_labels_match_existing_probes():
    from embedding_probe.common import PAGE_TYPES as probe_labels
    assert PAGE_TYPES == probe_labels


def test_vector_preflight_excludes_test_pages(fixture_data):
    from page_sequence.preflight import check
    config, _, _, _ = fixture_data
    report = check(config, sample_pages=120)
    assert report['passed']
    assert report['counts']['sampled_pages'] == 100
    assert report['counts']['stats_missing'] == 0


@pytest.mark.parametrize('problem', ['page_overlap', 'copy_group', 'order_tie', 'annotation_missing', 'annotation_unknown'])
def test_index_rejects_bad_splits_and_order(tmp_path, problem):
    docs = [{'library': 'a', 'doc': 'x', 'split': 'fit'}, {'library': 'a', 'doc': 'y', 'split': 'test'}]
    pages = [Page('a', 'x', 'p1', 0), Page('a', 'y', 'p2', 0)]
    annotations = []
    if problem == 'page_overlap':
        pages[1] = Page('a', 'y', 'p1', 0)
    elif problem == 'copy_group':
        for d in docs:
            d['copy_group'] = 'same-copy'
    elif problem == 'order_tie':
        pages.append(Page('a', 'x', 'p3', 0))
    elif problem == 'annotation_missing':
        annotations.append(Page('a', 'x', 'missing', 1, annotated_type='NormalPage'))
    else:
        pages[0] = Page('a', 'x', 'p1', 0, annotated_type='unknown')
    with pytest.raises((ValueError, __import__('sqlite3').IntegrityError)):
        build_index(tmp_path / 'index', docs, pages, annotations)
    assert not (tmp_path / 'index').exists()


def test_sampler_resume_and_coverage(fixture_data):
    _, index, _, _ = fixture_data
    a = WindowSampler(index, 12, seed=9)
    seen = set()
    while a.epoch == 0:
        seen.update(a._coverage())
    expected = {r for d in index.documents('fit') for r in range(d.first, d.first + d.length)}
    assert seen == expected
    for _ in range(17):
        a.draw()
    state = a.state_dict()
    expected_windows = [a.draw() for _ in range(30)]
    b = WindowSampler(index, 12, seed=987)
    b.load_state_dict(state)
    assert [b.draw() for _ in range(30)] == expected_windows
    for w in expected_windows:
        assert set(w.rows) <= expected
        pages = index.pages(w.rows)
        for p, q in zip(pages, pages[1:]):
            if p.doc_key == q.doc_key:
                assert q.order == p.order + 1
            else:
                assert p.order == 9 and q.order == 0
    assert a.counters['visible_join:same_volume'] > 0


def test_features_targets_padding_and_missing(fixture_data):
    _, index, features, dataset = fixture_data
    docs = index.documents('fit')
    rows = tuple(range(docs[0].first + 7, docs[0].first + 10)) + tuple(range(docs[1].first, docs[1].first + 3))
    example = dataset[Window(rows)]
    assert example['starts'].tolist() == [0, 0, 0, 1, 0, 0]
    assert example['pair_mask'].tolist() == [False, True, True, True, True, True]
    batch = collate_windows([example, dataset[Window(rows[:2])]])
    assert batch['attention_mask'][1].tolist() == [True, True, False, False, False, False]
    assert batch['weight'][0, 0] == .5
    assert batch['pair_weight'][0, 1] == .5
    assert batch['type'][1, -1] == IGNORE
    pages = index.pages(rows)
    x = features(pages)
    y = features(pages[2:3])
    for key in x:
        np.testing.assert_allclose(x[key][2:3], y[key])
    missing = copy.copy(pages[0])
    object.__setattr__(missing, 'page', 'missing')
    object.__setattr__(missing, 'path', 'public.256/missing.jpg')
    m = features([missing])
    assert not m['presence'].any()
    assert all(not x.any() for x in m.values())


def test_masked_losses_and_padding_are_invariant(fixture_data):
    _, index, features, dataset = fixture_data
    torch.set_num_threads(2)
    model = tiny_model(features.dims).eval()
    rows = tuple(range(index.documents('fit')[0].first, index.documents('fit')[0].first + 4))
    example = dataset[Window(rows)]
    alone = collate_windows([example])
    extra_rows = rows + tuple(range(rows[-1] + 1, rows[-1] + 4))
    larger = collate_windows([example, dataset[Window(extra_rows)]])
    with torch.no_grad():
        out = model(alone['features'], alone['attention_mask'])
        padded = model(larger['features'], larger['attention_mask'])
    torch.testing.assert_close(out['type_logits'][0], padded['type_logits'][0, :4], atol=1e-6, rtol=1e-5)
    # Unrelated examples cannot affect the first sequence; only padding is masked, not documents.
    larger['features']['image'][1] = 100
    with torch.no_grad():
        changed = model(larger['features'], larger['attention_mask'])
    torch.testing.assert_close(padded['type_logits'][0], changed['type_logits'][0])
    for key in ('type', 'placement'):
        alone[key][:] = IGNORE
    alone['pair_mask'][:] = False
    losses, denominators = task_losses(out, alone)
    assert losses['total'] == 0
    assert all(v == 0 for v in denominators.values())
    alone['type'][0, 0] = LABEL2ID['TitlePage']
    alone['gold'][0, 0] = True
    losses, denominators = task_losses(out, alone)
    assert denominators['gold_type'] == 1 and denominators['weak_type'] == 0
    assert torch.isfinite(losses['total']) and losses['total'] > 0


@pytest.mark.parametrize('n', [1, 2, 7, 8, 9, 15, 31, 100])
@pytest.mark.parametrize('strategy', ['center_logits', 'uniform_logits', 'uniform_probs', 'center_probs', 'central'])
def test_merging_page_and_pair_alignment(n, strategy):
    config = InferenceConfig(8, 7, strategy, 2)
    spans = window_spans(n, 8, 7)
    assert len(spans) == len(set(spans))
    merger = StreamMerger(n, 3, config)
    truth = np.arange(n, dtype=np.float32)
    for s, e in spans:
        t = truth[s:e]
        starts = t.copy()
        starts[0] = 9999  # Invalid first-position value MUST NOT pollute seams.
        merger.add(s, e, {'type_logits': np.stack([t, -t, 0 * t], -1),
                          'placement_logits': np.stack([-t, t, 0 * t], -1),
                          'starts_document_logits': starts,
                          'pair_mask': np.arange(e - s) > 0})
    result = merger.finalize()
    np.testing.assert_allclose(result['starts_document_probs'][1:], 1 / (1 + np.exp(-truth[1:])), rtol=2e-6)
    assert np.isnan(result['starts_document_probs'][0])
    assert not result['pair_mask'][0] and result['pair_mask'][1:].all()


def test_merger_rejects_missing_seam():
    merger = StreamMerger(4, 3, InferenceConfig(2, 1))
    for s, e in [(0, 2), (2, 4)]:
        merger.add(s, e, {'type_logits': np.zeros((2, 3)), 'placement_logits': np.zeros((2, 3)),
                          'starts_document_logits': np.zeros(2), 'pair_mask': np.array([False, True])})
    with pytest.raises(ValueError, match='adjacent-pair'):
        merger.finalize()


def test_unknown_document_inference_matches_indexed_inference(fixture_data):
    _, index, features, dataset = fixture_data
    torch.set_num_threads(2)
    model = tiny_model(features.dims)
    rows = tuple(range(81, 101))
    c = InferenceConfig(8, 4)
    expected = predict_stream(model, dataset, rows, c)
    # Erase every label/document/order field; none may affect inference.
    pages = [Page(p.library, '', p.page, 0, path=p.path) for p in index.pages(rows)]
    actual = tag_pages(model, features, pages, c)
    for key in ('type_probs', 'placement_probs', 'starts_document_probs', 'segment_ids'):
        np.testing.assert_allclose(expected[key], actual[key])


def test_manifests_reject_leakage_and_fragments(fixture_data, tmp_path):
    _, index, _, _ = fixture_data
    path = tmp_path / 'bad.jsonl'
    for rows in [[1, 2], [81, 91], [89, 91]]:
        path.write_text(json.dumps({'split': 'dev', 'id': 'bad', 'rows': rows}) + '\n')
        with pytest.raises(ValueError):
            load_streams(path, index, 'dev')


def test_full_stream_inference_and_evaluation(fixture_data):
    config, index, features, dataset = fixture_data
    torch.set_num_threads(2)
    model = tiny_model(features.dims)
    streams = load_streams(config['dev_manifest'], index, 'dev')
    stream = streams[0]
    result = predict_stream(model, dataset, stream['rows'], InferenceConfig(8, 4), 'cpu')
    assert model.training  # inference restores caller's mode
    assert len(result['type']) == 20 and result['processed_pages'] > 20
    pages = index.pages(stream['rows'])
    result['type'] = np.array([p.target()[0] for p in pages])
    result['placement'] = np.array([p.target()[2] for p in pages])
    result['starts_document_probs'] = np.array([np.nan] + [float(a.doc != b.doc) for a, b in zip(pages, pages[1:])])
    evaluator = Evaluator()
    evaluator.add(stream['id'], pages, result, {(d.library, d.doc): d for d in index.documents('dev')})
    report = evaluator.report()
    assert report['curated_type']['n'] == 8  # four annotated pages per document, counted once
    assert report['curated_type']['accuracy'] == 1
    assert report['starts']['all']['n'] == 19
    assert report['starts']['all']['f1'] == 1
    assert report['exact_spans']['tp'] == 2
    assert report['exact_stream_fraction'] == 1


@pytest.mark.parametrize('missing', ['curated_type', 'weak_type_agreement', 'placement_db_agreement', 'all'])
def test_evaluation_ignores_empty_task_masks(fixture_data, missing):
    from dataclasses import replace
    _, index, _, _ = fixture_data
    pages = index.pages(range(81, 101))
    fields = ({'annotated_type': ''} if missing == 'curated_type' else
              {'db_type': ''} if missing == 'weak_type_agreement' else
              {'placement': ''} if missing == 'placement_db_agreement' else
              {'annotated_type': '', 'db_type': '', 'placement': ''})
    pages = [replace(p, **fields) for p in pages]
    result = {'type': np.zeros(20, dtype=int), 'placement': np.zeros(20, dtype=int),
              'pair_mask': np.array([False] + [True] * 19),
              'starts_document_probs': np.array([np.nan] + [.1] * 19),
              'processed_pages': 20}
    evaluator = Evaluator()
    documents = {(d.library, d.doc): d for d in index.documents('dev')}
    evaluator.add('empty-task', pages, result, documents)
    report = evaluator.report()
    for group in ('curated_type', 'weak_type_agreement', 'placement_db_agreement'):
        if missing in (group, 'all'):
            assert report[group]['n'] == 0
            assert report[group]['accuracy'] is None
            assert report[group]['weighted_f1'] is None
    assert report['starts']['all']['n'] == 19


def test_training_resume_matches_uninterrupted(fixture_data, tmp_path):
    config, _, _, _ = fixture_data
    torch.set_num_threads(2)
    config['validate_every'] = 2
    uninterrupted = copy.deepcopy(config)
    uninterrupted['run_dir'] = str(tmp_path / 'uninterrupted')
    train(uninterrupted)
    train(config, stop_after=2)
    train(config, resume=tmp_path / 'run/latest.pt')
    a, sa = load_checkpoint(tmp_path / 'uninterrupted/latest.pt')
    b, sb = load_checkpoint(tmp_path / 'run/latest.pt')
    assert sa['step'] == sb['step'] == 4
    assert sa['sampler']['counters'] == sb['sampler']['counters']
    for k, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[k], rtol=0, atol=0)


def test_worker_training_resume_matches_uninterrupted(fixture_data, tmp_path):
    config, _, _, _ = fixture_data
    torch.set_num_threads(2)
    config['workers'] = 2
    config['validate_every'] = 2
    uninterrupted = copy.deepcopy(config)
    uninterrupted['run_dir'] = str(tmp_path / 'worker-uninterrupted')
    train(uninterrupted)
    train(config, stop_after=2)
    train(config, resume=tmp_path / 'run/latest.pt')
    a, sa = load_checkpoint(tmp_path / 'worker-uninterrupted/latest.pt')
    b, sb = load_checkpoint(tmp_path / 'run/latest.pt')
    assert sa['sampler']['counters'] == sb['sampler']['counters']
    for k, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[k], rtol=0, atol=0)


def test_validation_failure_keeps_resumable_checkpoint(fixture_data, tmp_path, monkeypatch):
    from page_sequence import runner
    config, _, _, _ = fixture_data
    config['validate_every'] = 2
    uninterrupted = copy.deepcopy(config)
    uninterrupted['run_dir'] = str(tmp_path / 'uninterrupted')
    train(uninterrupted)
    original = runner.evaluate_streams

    def fail_validation(*args, **kwargs):
        raise RuntimeError('simulated evaluator failure')

    monkeypatch.setattr(runner, 'evaluate_streams', fail_validation)
    with pytest.raises(RuntimeError, match='simulated evaluator failure'):
        train(config)
    _, failed = load_checkpoint(tmp_path / 'run/latest.pt')
    assert failed['step'] == 2
    assert failed['training']['pending_validation']['step'] == 2
    monkeypatch.setattr(runner, 'evaluate_streams', original)
    train(config, resume=tmp_path / 'run/latest.pt')
    a, sa = load_checkpoint(tmp_path / 'uninterrupted/latest.pt')
    b, sb = load_checkpoint(tmp_path / 'run/latest.pt')
    assert sa['sampler']['counters'] == sb['sampler']['counters']
    assert 'pending_validation' not in sb['training']
    for k, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[k], rtol=0, atol=0)


def test_plateau_changes_stage_and_saves_completed_state(fixture_data, tmp_path, monkeypatch):
    config, _, _, _ = fixture_data
    config.update(pretrain_steps=5, finetune_steps=2, validate_every=1)
    config['stopping'] = {'pretrain': {'patience': 1}}
    monkeypatch.setattr('page_sequence.runner.evaluate_streams', lambda *args: {
        'curated_type': {'weighted_f1': .5}})
    train(config)
    _, state = load_checkpoint(tmp_path / 'run/latest.pt')
    assert state['step'] == 4  # two pretraining updates, then two fine-tuning updates
    assert state['training']['stage'] == 'finetune'
    assert state['training']['stage_step'] == 2
    assert state['training']['finished']
    _, selected = load_checkpoint(tmp_path / 'run/pretrain_best.pt')
    assert selected['step'] == 1
