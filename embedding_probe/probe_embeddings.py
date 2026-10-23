"""Probes the frozen page embeddings for page type: classifiers trained on the annotated trn pages,
predicting the annotated tst pages, with a context window of K pages on each side (--context K;
K = 0 is the page alone).

Experiments (each a column of the predictions file):

  <model>              linear probe on the window's embeddings of one model
  <model> kNN          cosine k-nearest-neighbour vote over the trn pages, on the same window
  <text>+<image>       linear probe on the window's text and image embeddings together
  DB page type         the Kramerius page type as the prediction (a reference: how good the DB is)

Window features: the slots at offsets -K .. +K in order, per model the L2-normalized vector
standardized with the trn statistics of that slot; an empty slot (the window runs past the
document's start or end) is zeros, i.e. the mean, and for K > 0 a flag per offset says whether the
slot holds a page. For kNN the window is the concatenation of the L2-normalized slot vectors
(zeros where empty), normalized again.

The linear probe is multinomial logistic regression (full-batch L-BFGS, on the GPU when there is one)
with class-balanced loss weights. Its L2 weight decay is chosen per experiment on a holdout of the
trn documents (15%, never tst), then the probe is refitted on all trn pages.

Writes <work-dir>/k<K>/predictions/embeddings.parquet (key, label and one column per experiment,
tst pages) and embeddings.selection.json (the chosen weight decay and its holdout scores). Scores
are computed by report.py.

Example:
    python -m embedding_probe.probe_embeddings --context 0
"""
import argparse
import hashlib
import json
import logging
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import polars as pl

from embedding_probe.common import MODELS, PAGE_TYPES, WORK_DIR, evaluate, label_ids, setup_logging, window_ids
from embedding_probe.extract_vectors import Vectors

logger = logging.getLogger('PROBE EMBEDDINGS')
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3, 1e-2)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--context', type=int, default=0, help='Context window K: pages on each side.')
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--knn-k', type=int, default=10)
    parser.add_argument('--holdout-fraction', type=float, default=0.15)
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--only', nargs='*', default=None, help='Run only these experiments (by name).')
    return parser.parse_args(argv)


def stable_fraction(text: str) -> float:
    return int.from_bytes(hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest(), 'little') / 2 ** 64


def holdout_documents(pages: pl.DataFrame, fraction: float) -> np.ndarray:
    trn = (pages['split'] == 'trn').to_numpy()
    return trn & np.array([stable_fraction(f'{lib}/{doc}') < fraction
                           for lib, doc in pages.select('library', 'doc').iter_rows()])


def l2_normalized(vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def standardized(x: np.ndarray, present: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Standardizes with the mean/std of the present fit rows; missing rows become 0 (the mean)."""
    rows = fit_rows & present
    mean, std = x[rows].mean(axis=0), x[rows].std(axis=0)
    x = (x - mean) / np.maximum(std, 1e-6)
    x[~present] = 0
    return x.astype(np.float32)


def window_slots(pages: pl.DataFrame, vectors: Vectors, context: int):
    """Per offset -K .. +K: (raw vectors, present mask) of the window slot."""
    return [vectors.window(ids) for ids in window_ids(pages, vectors.kind, context)]


def linear_features(pages: pl.DataFrame, models: Sequence[Vectors], context: int, fit_rows: np.ndarray):
    parts, presence = [], []
    for vectors in models:
        for offset, (x, present) in zip(range(-context, context + 1), window_slots(pages, vectors, context)):
            if offset == 0:
                assert present.all(), f'{vectors.name}: an annotated page has no vector'
            parts.append(standardized(l2_normalized(x), present, fit_rows))
            if offset != 0 and vectors is models[0]:
                presence.append(present.astype(np.float32) - 0.5)
    return np.hstack(parts + ([np.stack(presence, axis=1)] if presence else []))


def knn_features(pages: pl.DataFrame, vectors: Vectors, context: int) -> np.ndarray:
    slots = [l2_normalized(x) * present[:, None] for x, present in window_slots(pages, vectors, context)]
    return l2_normalized(np.hstack(slots))


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return weights


def fit_linear(x: np.ndarray, y: np.ndarray, weight_decay: float, device: str, max_iter: int = 500):
    """Multinomial logistic regression, class-balanced, L2 on the weights; returns (W, b) as numpy."""
    import torch
    import torch.nn.functional as F

    n_classes = len(PAGE_TYPES)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(y, device=device)
    cw = torch.as_tensor(class_weights(y, n_classes), dtype=torch.float32, device=device)
    w = torch.zeros(x.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, history_size=20,
                                  line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(xt @ w + b, yt, weight=cw) + 0.5 * weight_decay * (w * w).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach().cpu().numpy(), b.detach().cpu().numpy()


def predict_linear(model, x: np.ndarray) -> np.ndarray:
    w, b = model
    return np.argmax(x @ w + b, axis=1)


def linear_probe(x: np.ndarray, y: np.ndarray, trn: np.ndarray, holdout: np.ndarray, tst: np.ndarray, device: str):
    """Chooses the weight decay on the holdout trn documents, refits on all trn pages, predicts tst."""
    fit = trn & ~holdout
    scores = {}
    for weight_decay in WEIGHT_DECAYS:
        model = fit_linear(x[fit], y[fit], weight_decay, device)
        pred = predict_linear(model, x[holdout])
        scores[weight_decay] = evaluate(names(y[holdout]), names(pred))['weighted_fscore']
    best = max(scores, key=scores.get)
    model = fit_linear(x[trn], y[trn], best, device)
    return predict_linear(model, x[tst]), {'weight_decay': best, 'holdout_weighted_fscore': scores}


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int, chunk: int = 2048):
    """Cosine kNN, votes weighted by similarity (x are L2-normalized)."""
    predictions = []
    for start in range(0, len(query_x), chunk):
        sims = query_x[start:start + chunk] @ train_x.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        top_sims = np.clip(np.take_along_axis(sims, top, axis=1), 0, None)
        votes = np.zeros((len(top), len(PAGE_TYPES)))
        np.add.at(votes, (np.repeat(np.arange(len(top)), k), train_y[top].ravel()), top_sims.ravel())
        predictions.append(votes.argmax(axis=1))
    return np.concatenate(predictions)


def names(ids: np.ndarray) -> List[str]:
    return [PAGE_TYPES[i] for i in ids]


def linear_experiments(models: Sequence[str]):
    """(name, models) of the linear-probe experiments: each model, and each text model with each image model."""
    text = [m for m in models if MODELS[m][0] == 'text']
    image = [m for m in models if MODELS[m][0] == 'image']
    return [(m, [m]) for m in models] + [(f'{t}+{i}', [t, i]) for t, i in product(text, image)]


def main(argv=None):
    args = parse_arguments(argv)
    out_dir = args.work_dir / f'k{args.context}' / 'predictions'
    setup_logging('PROBE EMBEDDINGS', args.work_dir / 'logs' / f'probe_embeddings.k{args.context}.txt')
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    pages = pl.read_parquet(args.work_dir / 'pages.parquet')
    y = label_ids(pages['label'].to_list())
    assert (y >= 0).all()
    trn = (pages['split'] == 'trn').to_numpy()
    tst = ~trn
    holdout = holdout_documents(pages, args.holdout_fraction)
    logger.info(f'K={args.context}: trn {trn.sum()} pages ({holdout.sum()} in the holdout documents), '
                f'tst {tst.sum()}; device {device}')
    vectors: Dict[str, Vectors] = {m: Vectors(args.work_dir, m) for m in args.models}

    wanted = (lambda name: args.only is None or name in args.only)
    predictions = {'key': pages['key'].to_numpy()[tst], 'label': pages['label'].to_numpy()[tst]}
    selection = {}
    if wanted('DB page type'):
        predictions['DB page type'] = pages['db_type'].fill_null('').to_numpy()[tst]
    for model in args.models:
        name = f'{model} kNN'
        if wanted(name):
            x = knn_features(pages, vectors[model], args.context)
            predictions[name] = names(knn_predict(x[trn], y[trn], x[tst], args.knn_k))
            logger.info(f'{name}: {x.shape[1]} features')
    for name, models in linear_experiments(args.models):
        if not wanted(name):
            continue
        started = time.time()
        x = linear_features(pages, [vectors[m] for m in models], args.context, trn)
        pred, selection[name] = linear_probe(x, y, trn, holdout, tst, device)
        predictions[name] = names(pred)
        logger.info(f'{name}: {x.shape[1]} features, weight decay {selection[name]["weight_decay"]}, '
                    f'holdout weighted F1 {selection[name]["holdout_weighted_fscore"]}, {time.time() - started:.0f}s')
        del x

    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({k: list(v) for k, v in predictions.items()}).write_parquet(out_dir / 'embeddings.parquet')
    (out_dir / 'embeddings.selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    logger.info(f'wrote {out_dir / "embeddings.parquet"}')


if __name__ == '__main__':
    main()
