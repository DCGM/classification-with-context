import argparse
import json
from pathlib import Path

import lmdb
import numpy as np
import pytest

from embeddings.text.page_stats import PageStats, to_sparse
from page_sequence.features import FeaturePipeline
from page_sequence.index import build_index
from page_sequence.schema import Page
from training_store.build import build_shard, plan_store, run, validate
from training_store.format import RECORD, atomic_json
from training_store.reader import TrainingStore


@pytest.fixture
def example(tmp_path):
    stats = PageStats()
    pages = [Page('lib', f'd{i // 3}', f'p{i}', i % 3 + 1,
                  f'{"public" if i < 3 else "private"}/p{i}', 'NormalPage',
                  'TitlePage' if i % 3 == 0 else '', 'left' if i % 2 else 'right') for i in range(6)]
    path = tmp_path / 'index.sqlite'
    build_index(path, [dict(library='lib', doc=f'd{i}', split='fit', volume='v', periodical='p')
                       for i in range(2)], pages)
    sources = {}
    for kind in ('image', 'text', 'stats'):
        stores = {}
        for scope in ('public', 'private'):
            dest = tmp_path / kind / scope
            dest.mkdir(parents=True)
            env = lmdb.open(str(dest), map_size=16 << 20)
            with env.begin(write=True) as txn:
                if kind == 'stats':
                    txn.put(b'__metadata__', json.dumps(stats.metadata()).encode())
                for i, p in enumerate(pages):
                    if kind == 'image' and scope != p.path.split('/')[0]:
                        continue
                    if i == 5:  # all modalities missing on a retained page
                        continue
                    if kind == 'stats':
                        x = np.zeros(len(stats.features), np.float32)
                        raw = to_sparse(x)
                    else:
                        x = (np.arange(768, dtype=np.float32) / 1000 + i + (9 if scope == 'private' else 0))
                        raw = x.astype('<f2').tobytes()
                    key = f'p{i}' if kind == 'image' else p.key
                    txn.put(key.encode(), raw)
            env.close()
            stores[scope] = str(dest)
        sources[kind] = dict(kind=kind, dim=len(stats.features) if kind == 'stats' else 768, stores=stores)
    config = {'index': str(path), 'features': {'sources': sources}}
    return config, pages


def prepare(tmp_path, config):
    output = tmp_path / 'store'
    output.mkdir()
    (output / 'shards').mkdir()
    plan = plan_store(output, config, config['features'], RECORD.itemsize * 4, 0)
    return output, plan


def publish(output, plan):
    reports = [build_shard(output, i, 2) for i in range(len(plan['shard_pages']))]
    report = validate(output, plan, 32)
    atomic_json(output / 'manifest.json', plan | {'complete': True, 'shards': reports})
    return report


def test_roundtrip_and_repeated_rows(tmp_path, example):
    config, pages = example
    output, plan = prepare(tmp_path, config)
    assert plan['shard_pages'] == [3, 3]
    report = publish(output, plan)
    assert report['fp16_rounded_reference_exact']
    reader = TrainingStore(output)
    records = reader.read_rows([6, 2, 1, 2, 4])
    assert records['row'].tolist() == [6, 2, 1, 2, 4]
    assert records['availability'][0] == 0
    expected = FeaturePipeline.from_config(config['features'])([pages[i-1] for i in [6, 2, 1, 2, 4]])
    for k, v in reader.features(records).items():
        assert v.dtype == np.float16
        np.testing.assert_array_equal(v, expected[k].astype(np.float16))
    assert reader.read_span(1, 1, 2)['row'].tolist() == [2, 3]
    assert reader.read_rows([]).shape == (0,)
    with pytest.raises(ValueError):
        reader.read_span(1, 2, 2)
    for rows in [[0], [7], [-1]]:
        with pytest.raises(KeyError):
            reader.read_rows(rows)
    reader.close()


def test_training_windows_switch_without_changing_inputs(tmp_path, example):
    from page_sequence.data import WindowDataset, collate_windows
    from page_sequence.index import PageIndex
    from page_sequence.schema import Window
    from training_store.dataset import StoreDataset
    config, _ = example
    output, plan = prepare(tmp_path, config)
    features = FeaturePipeline.from_config(config['features'])
    index = PageIndex(output / 'catalogue.sqlite')
    dataset = StoreDataset(index, features, output, allow_building=True)
    window = Window((6, 2, 1, 2, 4))
    before = dataset[window]
    assert before['storage_pages'] == {'nvme': 0, 'fallback': 5}
    for source in features.sources.values():
        for env in source._envs.values():
            env.close()
        source._envs = {}
    for i in range(len(plan['shard_pages'])):
        build_shard(output, i, 2)
    after = dataset[window]
    assert after['storage_pages'] == {'nvme': 5, 'fallback': 0}
    original = WindowDataset(index, features)[window]
    for k in ('type', 'gold', 'placement', 'starts', 'pair_mask', 'page_start', 'page_end'):
        np.testing.assert_array_equal(after[k], original[k])
        np.testing.assert_array_equal(after[k], before[k])
    for k in after['features']:
        np.testing.assert_array_equal(after['features'][k], original['features'][k].astype(np.float16))
        np.testing.assert_array_equal(after['features'][k], before['features'][k])
    batch = collate_windows([before, after])
    import torch
    assert all(v.dtype == torch.float16 for v in batch['features'].values())
    assert batch['storage_pages'] == {'nvme': 5, 'fallback': 5}


def test_training_rejects_mismatched_completion_marker(tmp_path, example):
    from page_sequence.index import PageIndex
    from page_sequence.schema import Window
    from training_store.dataset import StoreDataset
    config, _ = example
    output, plan = prepare(tmp_path, config)
    dataset = StoreDataset(PageIndex(output / 'catalogue.sqlite'),
                           FeaturePipeline.from_config(config['features']), output, allow_building=True)
    report = build_shard(output, 0, 2)
    atomic_json(output / 'shards/000000.done.json', report | {'build_id': 'wrong'})
    with pytest.raises(ValueError, match='completed shard'):
        dataset[Window((1, 2))]


def test_evaluation_relocates_published_store_without_reading_old_vectors(tmp_path, example, monkeypatch):
    import hashlib
    import shutil
    from page_sequence.runner import evaluation_dataset
    config, _ = example
    output, plan = prepare(tmp_path, config)
    publish(output, plan)
    state = {'config': config | {'training_store': {'path': str(output), 'allow_building': True}},
             'features': FeaturePipeline.from_config(config['features']).state_dict(),
             'provenance': {'index_sha256': plan['index_sha256'], 'training_store': {
                 'build_id': plan['build_id'],
                 'plan_sha256': hashlib.sha256((output / 'plan.json').read_bytes()).hexdigest()}}}
    moved = tmp_path / 'moved'
    shutil.copytree(output, moved)
    monkeypatch.setattr(FeaturePipeline, 'source_provenance', lambda self: pytest.fail('old LMDB accessed'))
    actual_config, dataset = evaluation_dataset(state, moved)
    assert actual_config['index'] == str(moved / 'catalogue.sqlite')
    assert not dataset.allow_building
    assert dataset.reader.read_rows([1, 6])['row'].tolist() == [1, 6]
    state['provenance']['training_store']['build_id'] = 'different'
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluation_dataset(state, moved)


def test_incomplete_and_shard_resume(tmp_path, example):
    config, _ = example
    output, plan = prepare(tmp_path, config)
    with pytest.raises(FileNotFoundError):
        TrainingStore(output)
    partial = output / 'shards/000000.partial'
    partial.write_bytes(b'interrupted')
    a = build_shard(output, 0, 2)
    assert not partial.exists()
    before = (output / 'shards/000000.bin').stat().st_mtime_ns
    assert build_shard(output, 0, 2) == a
    assert (output / 'shards/000000.bin').stat().st_mtime_ns == before
    with (output / 'shards/000000.bin').open('r+b') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError, match='completed shard changed'):
        build_shard(output, 0, 2)


def test_source_mismatch_blocks_publication(tmp_path, example):
    config, _ = example
    output, plan = prepare(tmp_path, config)
    for i in range(len(plan['shard_pages'])):
        build_shard(output, i, 2)
    shard = output / 'shards/000000.bin'
    data = np.fromfile(shard, RECORD)
    data['image'][0, 0] = 1
    data.tofile(shard)
    with pytest.raises(ValueError, match='differ|bytes changed'):
        validate(output, plan, 128)
    assert not (output / 'manifest.json').exists()


def test_full_cli_and_resume_guard(tmp_path, example):
    config, _ = example
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(config))
    output = tmp_path / 'complete'
    args = argparse.Namespace(config=str(cfg), sources=None, output=str(output), workers=1,
                              block_pages=2, shard_mib=1, pilot_documents=0, validation_pages=32)
    run(args)
    assert json.loads((output / 'status.json').read_text())['phase'] == 'complete'
    store = TrainingStore(output)
    assert len(store.read_rows([1, 6])) == 2
    store.close()
    (output / 'manifest.json').unlink()
    plan = json.loads((output / 'plan.json').read_text())
    plan['sources'] = {}
    atomic_json(output / 'plan.json', plan)
    with pytest.raises(ValueError, match='resume inputs'):
        run(args)
