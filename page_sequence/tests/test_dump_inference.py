from types import SimpleNamespace

import numpy as np
import torch

from page_sequence.dump_inference import ExportDataset, TileBatches, collate, pack_result
from page_sequence.inference import InferenceConfig, StreamMerger, predict_stream
from page_sequence.models import BertBackbone, PageTagger
from page_sequence.scan_borders import BoundaryFeatureDataset
from page_sequence.schema import IGNORE
from page_sequence.tests.test_auxiliary_edges import fixture
from training_store.format import DOCUMENT, RECORD


def test_batches_independent_documents_match_existing_inference(tmp_path):
    _, index, features, dataset = fixture(tmp_path)
    dataset = BoundaryFeatureDataset(dataset)
    docs = index.documents('fit')[:3]
    table = np.zeros(max(d.id for d in docs) + 1, DOCUMENT)
    offset = 0
    for d in reversed(docs):
        table[d.id] = d.first, d.length, 0, offset
        offset += d.length
    native = SimpleNamespace(documents=table, ids=np.array([d.id for d in docs]))
    settings = InferenceConfig(window_size=8, stride=6, batch_size=4)
    backbone = BertBackbone.create(pretrained=False, config=dict(hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=1,
        max_position_embeddings=16, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    torch.set_num_threads(1)
    model = PageTagger(backbone, dataset.model_dims, 'separate_mlp', auxiliary_edges=True).eval()
    exporter = ExportDataset(dataset)
    results = {}
    mergers = {}
    for tiles in TileBatches(native, settings, set()):
        batch = collate([exporter[t] for t in tiles])
        with torch.inference_mode():
            prediction = model(batch['features'], batch['attention_mask'], return_embeddings=True)
        for i, tile in enumerate(tiles):
            merger = mergers.setdefault(tile.document, StreamMerger(tile.length, model.num_types,
                                                                      settings, True, True))
            merger.add(tile.start, tile.end,
                       {k: v[i, :tile.end-tile.start].numpy() for k, v in prediction.items()})
            if tile.end == tile.length:
                results[tile.document] = merger.finalize()
    for d in docs:
        starts, ends = np.zeros(d.length), np.zeros(d.length)
        starts[0], ends[-1] = 1, 1
        expected = predict_stream(model, dataset, tuple(range(d.first, d.first + d.length)),
                                  settings, document_starts=starts, document_ends=ends,
                                  collect_embeddings=True, collect_window_stats=True)
        for key in ('type_probs', 'placement_probs', 'page_start_probs', 'page_end_probs',
                    'starts_document_probs', 'embeddings', 'type_window_js'):
            np.testing.assert_allclose(results[d.id][key], expected[key], atol=2e-6)
        np.testing.assert_array_equal(results[d.id]['window_count'], expected['window_count'])
    assert not list(TileBatches(native, settings, {0}))


def test_packed_metadata_missing_labels_and_fp16_roundtrip(tmp_path):
    from page_sequence.dump_inference import RECORD as OUTPUT_RECORD
    source = np.zeros(2, RECORD)
    source['row'], source['document'], source['weak'], source['gold'] = [10, 11], 7, [1, IGNORE], [IGNORE, 2]
    source['availability'] = [0, 5]
    probabilities = np.zeros((2, 39), np.float32)
    probabilities[:, [1, 2]] = [.8, .2]
    result = dict(type_probs=probabilities, placement_probs=np.full((2, 3), 1/3),
                  type=np.array([1, 1]), placement=np.zeros(2), starts_document_probs=[np.nan, .1],
                  pair_mask=[False, True], page_start_probs=[.9, .1], page_end_probs=[.1, .9],
                  window_count=[1, 2], type_window_js=[0, .01], embeddings=np.ones((2, 16)))
    pagination = np.array([(5, 1), (0, 0)], dtype=[('value', '<u4'), ('system', 'u1')])
    output = pack_result(source, pagination, result)
    assert output.dtype == OUTPUT_RECORD
    assert output['position'].tolist() == [0, 1]
    assert output['document_length'].tolist() == [2, 2]
    assert output['gold_disagrees'].tolist() == [False, True]
    assert output['weak_disagrees'].tolist() == [False, False]
    assert np.isnan(output['gold_probability'][0])
    assert np.isnan(output['weak_probability'][1])
    np.testing.assert_allclose(output['gold_probability'][1], .2)
    path = tmp_path / 'records.npy'
    np.save(path, output)
    restored = np.load(path, mmap_mode='r')
    np.testing.assert_array_equal(restored['row'], source['row'])
    assert not restored['pair_mask'][0] and np.isnan(restored['starts_document_probs'][0])
