import copy
import json
from pathlib import Path
import shutil

import lmdb
import numpy as np
import pytest

from embeddings.text.page_stats import PageStats, to_sparse
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.schema import Window
from training_store.dataset import StoreDataset
from training_store.format import SPARSE_VERSION
from training_store.reader import TrainingStore
from training_store.refresh import run
from training_store.tests.test_store import example, prepare, publish


@pytest.mark.parametrize('interrupted', [False, True])
def test_refresh_preserves_raw_stats_and_metadata_and_uses_new_text(tmp_path, example, monkeypatch, interrupted):
    config, pages = example
    vector = np.zeros(len(PageStats.features), np.float32)
    vector[[0, 4, 1000]] = [2, .125, 500]
    raw = to_sparse(vector)
    env = lmdb.open(config['features']['sources']['stats']['stores']['public'])
    with env.begin(write=True) as txn:
        txn.put(pages[1].key.encode(), raw)
    env.close()
    base, plan = prepare(tmp_path, config)
    publish(base, plan)
    base_manifest = json.loads((base / 'manifest.json').read_text())
    (base / 'checksums.json').write_text(json.dumps({
        f"shards/{r['shard']:06d}.bin": r['sha256'] for r in base_manifest['shards']}))
    sources = copy.deepcopy(config['features'])
    for scope, path in sources['sources']['text']['stores'].items():
        dest = tmp_path / f'new-text-{scope}'
        shutil.copytree(path, dest)
        sources['sources']['text']['stores'][scope] = str(dest)
        env = lmdb.open(str(dest))
        with env.begin(write=True) as txn:
            for i, page in enumerate(pages[:-1]):
                value = (np.arange(768) / 300 + 10 + i + (20 if scope == 'private' else 0)).astype('<f2')
                txn.put(page.key.encode(), value.tobytes())
            if scope == 'public':
                txn.delete(pages[2].key.encode())  # private fallback must be used
        env.close()
    output = tmp_path / 'new-store'
    if interrupted:
        from training_store import refresh
        original = refresh.ingest

        def stop_after_text(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('simulated interruption after durable text stage')

        monkeypatch.setattr(refresh, 'ingest', stop_after_text)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            run(base, output, sources, sample_pages=32)
        assert not (output / 'manifest.json').exists()
        monkeypatch.setattr(refresh, 'ingest', original)
    if interrupted:
        run(base, output, sources, sample_pages=32)
    else:
        from training_store.refresh_pipeline import run as run_pipeline
        work, prepared, ssd = tmp_path / 'workflow', tmp_path / 'relocation', tmp_path / 'ssd-store'
        prepared.mkdir()
        (prepared / 'status.json').write_text(json.dumps({'phase': 'complete'}))
        source_config = tmp_path / 'sources.json'
        source_config.write_text(json.dumps(sources))
        run_pipeline(dict(work=str(work), relocation_journal=str(prepared), base=str(base),
                          nvme=str(output), ssd=str(ssd), sources=str(source_config), validation_pages=32))
        assert (ssd / 'manifest.json').read_bytes() == (output / 'manifest.json').read_bytes()
        assert output.is_dir() and not output.is_symlink()
        assert ssd.is_dir() and not ssd.is_symlink()
        assert json.loads((work / 'pipeline-status.json').read_text())['phase'] == 'complete'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['version'] == SPARSE_VERSION
    assert manifest['complete']
    assert not (output / 'scratch').exists()
    assert (output / 'catalogue.sqlite').read_bytes() == (base / 'catalogue.sqlite').read_bytes()
    assert (output / 'documents.npy').read_bytes() == (base / 'documents.npy').read_bytes()
    before, after = TrainingStore(base), TrainingStore(output)
    records = after.read_rows([6, 2, 3, 2, 1])
    old = before.read_rows([6, 2, 3, 2, 1])
    assert records['stats'].tolist() == [b'', raw, b'', raw, b'']
    assert records['availability'][0] == 0
    assert records['availability'][-1] == 7  # present all-zero sparse vector
    for name in ('image', 'row', 'document', 'order', 'weak', 'gold', 'placement'):
        np.testing.assert_array_equal(records[name], old[name])
    assert not np.array_equal(records['text'][1], old['text'][1])
    assert after.read_rows([]).shape == (0,)
    assert after.read_span(1, 0, 3)['row'].tolist() == [1, 2, 3]
    dataset = StoreDataset(PageIndex(output / 'catalogue.sqlite'),
                           FeaturePipeline.from_config(sources), output)
    result = dataset[Window((6, 2, 3, 2, 1))]
    expected = FeaturePipeline.from_config(sources)([pages[i-1] for i in (6, 2, 3, 2, 1)])
    for name in result['features']:
        np.testing.assert_array_equal(result['features'][name], expected[name].astype(np.float16))
    with pytest.raises(ValueError, match='published'):
        StoreDataset(PageIndex(output / 'catalogue.sqlite'),
                     FeaturePipeline.from_config(sources), output, allow_building=True)
    run(base, output, sources, sample_pages=32)  # completed build is not rewritten
    before.close()
    after.close()
    dataset.reader.close()
