import copy
import csv
import json

import numpy as np
import pytest
import torch

from page_sequence.pagination import RECORD, POLICY, PaginationDataset, build, candidate, encode
from page_sequence.manifests import file_sha256
from page_sequence.models import PageTagger, PageBackbone, from_specification
from page_sequence.runner import train, evaluation_dataset, load_checkpoint, make_dataset
from page_sequence.features import FeaturePipeline
from page_sequence.index import PageIndex
from page_sequence.schema import Window
from page_sequence.smoke import prepare_fixture
from page_sequence.data import collate_windows, to_device


@pytest.mark.parametrize('text,expected', [
    ('12', (12, 1)), (' XIV ', (14, 2)), ('iv', (4, 2)), ('55 [58]', (55, 1)),
    ('[12]', (0, 0)), ('[12a]', (0, 0)), ('4,[5]', (0, 0)), ('5,6', (0, 0)),
    ('12-14', (0, 0)), ('12a', (0, 0)), ('IIII', (0, 0)), ('0', (0, 0)),
    ('', (0, 0)), ('[12', (0, 0)), ('12 [13] junk', (0, 0)),
])
def test_candidate(text, expected):
    assert candidate(text) == expected


def test_encoding():
    r = np.array([(1, 1), (2, 1), (450, 1), (14, 2), (0, 0), (20000, 1)], RECORD)
    x = encode(r)
    assert x.dtype == np.float16 and np.isfinite(x).all()
    assert (x >= 0).all() and (x <= 1).all()
    assert x[0, 0] < x[1, 0] < x[2, 0]
    assert x[0, 1] == 1 and x[1, 1] == 0
    np.testing.assert_array_equal(x[3, 2:], [0, 1, 1])
    assert not x[4].any() and x[5, 0] == 1


def sidecar(config, path):
    index = PageIndex(config['index'])
    n = index.db.execute('SELECT MAX(id) FROM pages').fetchone()[0] + 1
    r = np.zeros(n, RECORD)
    r['value'][1:] = np.arange(1, n)
    r['system'][1:] = 1
    np.save(path, r)
    path.with_suffix('.json').write_text(json.dumps({'policy': POLICY, 'index_sha256': index.fingerprint(),
                                                   'sha256': file_sha256(path), 'rows': n}))
    return index


def test_window_alignment_mask_and_model_roundtrip(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=2)
    path = tmp_path / 'pagination.npy'
    index = sidecar(config, path)
    config['pagination'] = {'path': str(path), 'cap': 10000}
    features = FeaturePipeline.from_config(config['features'])
    dataset = make_dataset(index, features, config)
    example = dataset[Window((3, 1, 3))]
    expected = encode(dataset.records[[3, 1, 3]])
    np.testing.assert_array_equal(example['features']['pagination'], expected)
    batch = to_device(collate_windows([example, dataset[Window((2,))]]), 'cpu', torch.float32)
    model = PageTagger(PageBackbone(16), dataset.model_dims, 'separate_mlp')
    model.eval()
    restored = from_specification(model.specification())
    restored.load_state_dict(model.state_dict())
    restored.eval()
    a = model(batch['features'], batch['attention_mask'])
    b = restored(batch['features'], batch['attention_mask'])
    torch.testing.assert_close(a['type_logits'], b['type_logits'])
    # Missing pagination must contribute zero even with biased branch layers.
    masked = copy.deepcopy(batch['features'])
    masked['pagination'][..., 4] = 0
    original = model(masked, batch['attention_mask'])['type_logits']
    masked['pagination'][..., :4] = 99
    torch.testing.assert_close(original, model(masked, batch['attention_mask'])['type_logits'])


def test_training_resume_and_evaluation_identity(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=2)
    path = tmp_path / 'pagination.npy'
    sidecar(config, path)
    config['pagination'] = {'path': str(path), 'cap': 10000}
    config['fusion'] = 'separate_mlp'
    train(config, stop_after=1)
    checkpoint = __import__('pathlib').Path(config['run_dir']) / 'latest.pt'
    _, state = load_checkpoint(checkpoint)
    _, dataset = evaluation_dataset(state)
    assert dataset.model_dims['pagination'] == 5
    train(config, resume=checkpoint)
    # A different cap changes the exact feature contract.
    modified = copy.deepcopy(state)
    modified['config']['pagination']['cap'] = 100
    with pytest.raises(ValueError, match='pagination sidecar'):
        evaluation_dataset(modified)


def test_streaming_builder_and_missing_coverage(tmp_path):
    config = prepare_fixture(tmp_path / 'fixture', steps=2)
    index = PageIndex(config['index'])
    rows = list(index.db.execute("SELECT d.library,d.doc,p.page,p.ord FROM pages p JOIN documents d ON d.id=p.document WHERE d.split IN ('fit','dev') ORDER BY d.id,p.ord"))
    source = tmp_path / 'source.csv'
    with source.open('w', newline='') as f:
        writer = csv.writer(f)
        for lib, doc, page, order in rows:
            writer.writerow([lib, '', '', doc, page, '', '', '', order, '12' if order % 2 else '[12]', '', '', ''])
    output = tmp_path / 'built.npy'
    build(config['index'], source, output)
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['counts']['matched'] == len(rows)
    assert report['counts']['arabic'] > 0 and report['counts']['unavailable'] > 0
    with source.open('w', newline='') as f:
        csv.writer(f).writerow(['none', '', '', 'none', 'none', '', '', '', 0, '12', '', '', ''])
    with pytest.raises(ValueError, match='coverage mismatch'):
        build(config['index'], source, tmp_path / 'bad.npy')
    assert not (tmp_path / 'bad.npy').exists()


def test_optional_inference_and_forkserver_loading(tmp_path):
    from dataclasses import replace
    from page_sequence.inference import tag_pages, InferenceConfig
    from page_sequence.loading import training_loader
    from page_sequence.sampling import WindowSampler
    config = prepare_fixture(tmp_path / 'fixture', steps=2)
    path = tmp_path / 'pagination.npy'
    index = sidecar(config, path)
    config.update(pagination={'path': str(path)}, workers=2, worker_start_method='forkserver')
    features = FeaturePipeline.from_config(config['features'])
    dataset = make_dataset(index, features, config)
    loader, batches = training_loader(dataset, WindowSampler(index, 12), config)
    batches.batches = 2
    try:
        for batch in loader:
            for i, rows in enumerate(batch['rows']):
                np.testing.assert_array_equal(batch['features']['pagination'][i, :len(rows)].numpy(),
                                              encode(dataset.records[list(rows)]))
    finally:
        if loader._iterator:
            loader._iterator._shutdown_workers()
    pages = [replace(p, pagenum='12' if i % 2 else '[12]')
             for i, p in enumerate(index.pages(tuple(range(1, 11))))]
    model = PageTagger(PageBackbone(16), dataset.model_dims, 'separate_mlp')
    prediction = tag_pages(model, features, pages, InferenceConfig(window_size=8, stride=4))
    assert len(prediction['type']) == len(pages)


def test_checksum_streams_without_read_bytes(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    path = tmp_path / 'large.bin'
    data = b'pagination' * 200000
    path.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError('whole-file read')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
