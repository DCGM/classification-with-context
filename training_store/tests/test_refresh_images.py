import copy
import json
import shutil

import lmdb
import numpy as np
import pytest

from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.schema import Window
from training_store.dataset import StoreDataset
from training_store.reader import TrainingStore
from training_store.refresh import run as refresh_text
from training_store.refresh_images import run
from training_store.tests.test_store import example, prepare, publish  # noqa: F401


@pytest.mark.parametrize('interrupted', [False, True])
def test_image_refresh_replaces_only_images_without_text_or_stats_lmdbs(tmp_path, example, monkeypatch,
                                                                       interrupted):
    config, pages = example
    v1, plan = prepare(tmp_path, config)
    publish(v1, plan)
    manifest = json.loads((v1 / 'manifest.json').read_text())
    (v1 / 'checksums.json').write_text(json.dumps({
        f"shards/{r['shard']:06d}.bin": r['sha256'] for r in manifest['shards']}))
    base = tmp_path / 'sparse'
    refresh_text(v1, base, config['features'], sample_pages=32)
    stores = {}
    for scope, path in config['features']['sources']['image']['stores'].items():
        dest = tmp_path / f'new-image-{scope}' / 'lmdb'
        shutil.copytree(path, dest)
        env = lmdb.open(str(dest))
        with env.begin(write=True) as txn:
            for i, page in enumerate(pages):
                if page.path.split('/')[0] == scope and i not in (1, 5):
                    txn.put(f'p{i}'.encode(), (np.arange(768) / 500 - 3 - i).astype('<f2').tobytes())
            txn.delete(b'p1')  # an image the old source had
            if scope == 'private':
                txn.put(b'p5', (np.arange(768) / 700 + 7).astype('<f2').tobytes())  # one it lacked
        env.close()
        stores[scope] = str(dest)
    hidden = {}
    for kind in ('text', 'stats'):
        hidden[kind] = tmp_path / f'hidden-{kind}'
        shutil.move(tmp_path / kind, hidden[kind])
    output = tmp_path / 'dino-orig'
    if interrupted:
        from training_store import refresh_images
        original = refresh_images.finalize

        def stop(*args, **kwargs):
            raise RuntimeError('simulated interruption before finalizing')

        monkeypatch.setattr(refresh_images, 'finalize', stop)
        with pytest.raises(RuntimeError, match='simulated'):
            run(base, output, stores, sample_pages=32)
        assert not (output / 'manifest.json').exists()
        monkeypatch.setattr(refresh_images, 'finalize', original)
    run(base, output, stores, sample_pages=32)
    for kind, path in hidden.items():
        shutil.move(path, tmp_path / kind)
    new = json.loads((output / 'manifest.json').read_text())
    old = json.loads((base / 'manifest.json').read_text())
    assert new['complete'] and not (output / 'scratch').exists()
    assert new['feature_state'] == old['feature_state']
    assert new['reference_sources']['text'] == old['reference_sources']['text']
    assert new['reference_sources']['stats'] == old['reference_sources']['stats']
    assert new['reference_sources']['image']['stores']['private']['path'] == str(
        (tmp_path / 'new-image-private' / 'lmdb').resolve())
    report = json.loads((output / 'validation-report.json').read_text())
    assert report['image_change']['lost'] == 1 and report['image_change']['gained'] == 1
    before, after = TrainingStore(base), TrainingStore(output)
    rows = [1, 2, 3, 4, 5, 6]
    a, b = after.read_rows(rows), before.read_rows(rows)
    for name in ('text', 'stats', 'row', 'document', 'order', 'weak', 'gold', 'placement'):
        assert a[name].tolist() == b[name].tolist() if name == 'stats' else np.array_equal(a[name], b[name])
    np.testing.assert_array_equal(a['availability'] & 6, b['availability'] & 6)
    assert [bool(x & 1) for x in a['availability']] == [True, False, True, True, True, True]
    sources = copy.deepcopy(config['features'])
    sources['sources']['image']['stores'] = stores
    features = FeaturePipeline.from_config(sources)
    dataset = StoreDataset(PageIndex(output / 'catalogue.sqlite'), features, output)
    result = dataset[Window((6, 2, 3, 2, 1))]
    expected = features([pages[i - 1] for i in (6, 2, 3, 2, 1)])
    for name in result['features']:
        np.testing.assert_array_equal(result['features'][name], expected[name].astype(np.float16))
    assert dataset.vector_provenance(features) == new['reference_sources']
    run(base, output, stores, sample_pages=32)  # a published store is not rewritten
    before.close(), after.close(), dataset.reader.close()
