"""Evaluates the cheap text cues (text_cues.py) on the annotated pages, alone and added to the page
embeddings, with a context window of K pages on each side (--context K).

  text cues                 gradient boosting on the cue vectors of the window slots
  <embeddings> +cues        the linear probe of probe_embeddings.py on the window's embeddings plus the
                            standardized cue vectors of the window slots (default: mmBERT+DINOv2)

Window slots, empty slots, presence flags, the trn/tst split, the holdout for the weight decay and
--train-labels are as in probe_embeddings.py and text_baselines.py, so the scores compare directly
with '<embeddings>' and 'layout stats' there.

PRIVATE TEXT: as in text_baselines.py, the private text LMDB is read only when --private-text-lmdb is
given; texts and cue vectors stay in memory, and only counts and predicted labels are written or
logged. Without it only pages with public text are used and the output is named *.public_only.

With --alto-lmdb (the ALTO LMDBs of both storages, same keys as the text LMDBs) the cues are computed
on the ALTO line texts sorted top to bottom instead of the plain text, and the ALTO line features
(ocr_layout.LINE_FEATURES) are appended to each page's cue vector; the experiments are then named
'text cues +ALTO' and '<embeddings> +cues +ALTO'. The XML is parsed in worker processes that return
only the feature vectors.

Writes <work-dir>/k<K>/predictions/text_cues[.alto][.db_labels][.public_only].parquet. Scores: report.py.

Example:
    python -m embedding_probe.probe_text_cues --context 0 --private-text-lmdb \\
        /run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump.private/lmdb
    python -m embedding_probe.probe_text_cues --context 0 --alto-lmdb \
        /run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump/alto \
        /run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump.private/alto
"""
import argparse
import logging
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lmdb
import numpy as np
import polars as pl

from embedding_probe.common import MODELS, PUBLIC_TEXT_LMDB, WORK_DIR, setup_logging, window_ids
from embedding_probe.extract_vectors import Vectors
from embedding_probe.probe_embeddings import (TRAIN_LABEL_SUFFIX, holdout_documents, linear_features, linear_probe,
                                              names, standardized, training_labels)
from embedding_probe.text_baselines import presence_flags, read_texts, slot_rows
from embedding_probe.ocr_layout import LAYOUT_XML_FEATURES, LINE_FEATURES, layout_vector, ocr_text_lines
from embedding_probe.text_cues import cue_matrix, cue_vector

logger = logging.getLogger('PROBE TEXT CUES')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--context', type=int, default=0, help='Context window K: pages on each side.')
    parser.add_argument('--embeddings', nargs='+', default=['mmBERT', 'DINOv2'], choices=list(MODELS),
                        help='The embeddings the cues are added to.')
    parser.add_argument('--train-labels', choices=('annotated', 'db'), default='annotated')
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--public-text-lmdb', type=Path, default=PUBLIC_TEXT_LMDB)
    parser.add_argument('--private-text-lmdb', type=Path, default=None,
                        help='The private text LMDB; read only by this script, never to be looked at.')
    parser.add_argument('--alto-lmdb', type=Path, nargs='+', default=None,
                        help='ALTO LMDBs (public and private); cues from ALTO lines plus ALTO line features. '
                             'Read only by this script, never to be looked at.')
    parser.add_argument('--holdout-fraction', type=float, default=0.15)
    parser.add_argument('--device', default=None)
    return parser.parse_args(argv)


def window_cues(cues: np.ndarray, rows, fit_rows: np.ndarray) -> np.ndarray:
    """Per slot the standardized cue vector (zeros where the slot is empty), concatenated."""
    parts = []
    for r in rows:
        x = np.zeros((len(r), cues.shape[1]), dtype=np.float32)
        x[r >= 0] = cues[r[r >= 0]]
        parts.append(standardized(x, r >= 0, fit_rows))
    return np.hstack(parts)


_alto_envs = []
LINE_COLUMNS = [LAYOUT_XML_FEATURES.index(n) for n in LINE_FEATURES]


def _open_alto(paths):
    _alto_envs.extend(lmdb.open(str(p), readonly=True, lock=False, readahead=False) for p in paths)


def _alto_vectors(keys):
    """(found keys, their cue vectors on the ALTO lines + ALTO line features), in a worker process."""
    found, vectors = [], []
    for key in keys:
        value = None
        for env in _alto_envs:
            with env.begin() as txn:
                value = txn.get(key.encode('utf-8'))
            if value is not None:
                break
        if value is None:
            continue
        xml = zlib.decompress(value)
        layout = layout_vector(xml)
        found.append(key)
        vectors.append(cue_vector('\n'.join(ocr_text_lines(xml))) + [layout[i] for i in LINE_COLUMNS])
    return found, vectors


def alto_features(keys, paths, workers: int = None):
    """key -> row, matrix of cue vectors on the ALTO lines + ALTO line features, for the keys with ALTO."""
    workers = workers or min(os.cpu_count() or 1, 12)
    chunks = [keys[i:i + 1000] for i in range(0, len(keys), 1000)]
    found, vectors = [], []
    with ProcessPoolExecutor(workers, initializer=_open_alto, initargs=(paths,)) as pool:
        for f, v in pool.map(_alto_vectors, chunks):
            found += f
            vectors += v
    return found, np.array(vectors, dtype=np.float32)


def main(argv=None):
    args = parse_arguments(argv)
    alto = args.alto_lmdb is not None
    suffix = (('.alto' if alto else '') + ('.db_labels' if args.train_labels == 'db' else '')
              + ('' if args.private_text_lmdb or alto else '.public_only'))
    setup_logging('PROBE TEXT CUES', args.work_dir / 'logs' / f'probe_text_cues.k{args.context}{suffix}.txt')
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    pages = pl.read_parquet(args.work_dir / 'pages.parquet')
    slots = window_ids(pages, 'text', args.context)
    keys = sorted({k for slot in slots for k in slot if k is not None})
    started = time.time()
    if alto:
        unique, cues = alto_features(keys, args.alto_lmdb)
    else:
        texts = read_texts(keys, args.public_text_lmdb, args.private_text_lmdb)
        unique = sorted(texts)
        cues = cue_matrix([texts[k] for k in unique])
        del texts
    logger.info(f'K={args.context}: cues of {len(unique)} of {len(keys)} window pages, {cues.shape[1]} each, '
                f'{time.time() - started:.0f}s')
    rows = slot_rows(slots, {k: i for i, k in enumerate(unique)})
    keep = rows[args.context] >= 0
    pages, rows = pages.filter(pl.Series(keep)), [r[keep] for r in rows]
    y, trn, tst = training_labels(pages, args.train_labels)
    holdout = holdout_documents(pages, args.holdout_fraction) & trn
    name_suffix = (' +ALTO' if alto else '') + TRAIN_LABEL_SUFFIX[args.train_labels]
    logger.info(f'trn {trn.sum()} ({holdout.sum()} holdout), tst {tst.sum()} pages with {"ALTO" if alto else "text"}; '
                f'device {device}')

    predictions = {'key': pages['key'].to_numpy()[tst], 'label': pages['label'].to_numpy()[tst]}
    flags = presence_flags(rows, args.context)
    x = np.hstack([window_cues(cues, rows, trn), flags])
    model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, early_stopping=False,
                                           random_state=0).fit(x[trn], y[trn])
    predictions['text cues' + name_suffix] = names(model.predict(x[tst]))
    logger.info(f'text cues: {x.shape[1]} features, {time.time() - started:.0f}s')

    vectors = [Vectors(args.work_dir, m) for m in args.embeddings]
    x = np.hstack([linear_features(pages, vectors, args.context, trn), window_cues(cues, rows, trn)])
    name = '+'.join(args.embeddings) + ' +cues' + name_suffix
    pred, selection = linear_probe(x, y, trn, holdout, tst, device)
    predictions[name] = names(pred)
    logger.info(f'{name}: {x.shape[1]} features, weight decay {selection["weight_decay"]}, '
                f'{time.time() - started:.0f}s')

    out_dir = args.work_dir / f'k{args.context}' / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'text_cues{suffix}.parquet'
    pl.DataFrame({k: list(v) for k, v in predictions.items()}).write_parquet(out_path)
    logger.info(f'wrote {out_path}')


if __name__ == '__main__':
    main()
