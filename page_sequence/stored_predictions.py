"""Run a model once over the evaluation streams, store all its outputs, and derive reports from them.

`store_predictions` writes, per stream, every merged output of `inference.predict_stream`: the probabilities of all
heads (type, placement, pairwise starts, page start, page end), the predicted labels, the merge weights, the
window statistics (window count, type disagreement between windows) and the page embeddings (float16), with the
stream ID and its catalogue rows. `complete.json` marks a finished directory (written last, under a temporary
name until then) and records the model, data and inference behind it.

`evaluate_stored` recomputes `Evaluator` reports from such a directory with only the catalogue: no model and no
GPU, so any later statistic is computed without evaluating the system again.
"""
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .evaluation import Evaluator, border_result
from .inference import predict_stream

COMPLETE = 'complete.json'


def store_predictions(model, dataset, streams, inference, device, directory, provide_document_metadata=False,
                      metadata=None):
    directory = Path(directory)
    if (directory / COMPLETE).exists():
        raise FileExistsError(directory)
    partial = directory.with_name(directory.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    if directory.exists():
        raise FileExistsError(f'{directory} exists without {COMPLETE}; inspect before replacing it')
    partial.mkdir(parents=True)
    start = time.perf_counter()
    for i, stream in enumerate(streams):
        supplied = {}
        if provide_document_metadata:
            from .data import page_edge_targets
            targets = page_edge_targets(dataset.index, stream['rows'])
            supplied = dict(document_starts=targets['page_start'], document_ends=targets['page_end'])
        result = predict_stream(model, dataset, stream['rows'], inference, device, **supplied,
                                collect_embeddings=True, collect_window_stats=True)
        result['embeddings'] = result['embeddings'].astype(np.float16)
        # Filenames use ordinal positions, not untrusted manifest IDs or source paths.
        np.savez_compressed(partial / f'{i:06d}.npz', stream_id=str(stream['id']),
                            rows=np.asarray(stream['rows'], np.int64), **result)
    seconds = time.perf_counter() - start
    (partial / COMPLETE).write_text(json.dumps(dict(
        metadata or {}, streams=len(streams), stream_ids=[str(s['id']) for s in streams], seconds=seconds,
        document_metadata_provided=bool(provide_document_metadata), embeddings_dtype='float16'), indent=2) + '\n')
    partial.rename(directory)
    return directory


def load_stream(directory, i, stream=None):
    """The stored result of the i-th stream, as `predict_stream` returned it (embeddings in float16)."""
    with np.load(Path(directory) / f'{i:06d}.npz', allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    stream_id, rows = str(result.pop('stream_id')), result.pop('rows')
    if stream is not None and (stream_id != str(stream['id']) or not np.array_equal(rows, stream['rows'])):
        raise ValueError('stored predictions belong to other streams')
    return {k: v.item() if v.ndim == 0 else v for k, v in result.items()}


def evaluate_stored(directory, index, streams, threshold, border_sources=('pair',)):
    """One `Evaluator` report per border source, from stored predictions of exactly these streams."""
    directory = Path(directory)
    meta = json.loads((directory / COMPLETE).read_text())
    if meta['stream_ids'] != [str(s['id']) for s in streams]:
        raise ValueError('stored predictions cover other streams')
    evaluators = {source: Evaluator(threshold) for source in border_sources}
    for i, stream in enumerate(streams):
        result = load_stream(directory, i, stream)
        pages = index.pages(stream['rows'])
        documents = {(d.library, d.doc): d for d in index.documents_for_rows(stream['rows'])}
        for source, evaluator in evaluators.items():
            evaluator.add(stream['id'], pages, border_result(result, source), documents,
                          stream['censored_left'], stream['censored_right'])
    reports = {}
    for source, evaluator in evaluators.items():
        report = reports[source] = evaluator.report()
        report['seconds'] = meta['seconds']
        report['unique_pages_per_second'] = sum(s['pages'] for s in report['streams']) / meta['seconds']
    return reports
