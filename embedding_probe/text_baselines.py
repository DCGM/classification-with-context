"""Text-only baselines for the embedding probes: what simple analysis of the OCR text gets, with the
same context window of K pages on each side as probe_embeddings.py (--context K).

  layout stats     gradient boosting on 17 line-structure statistics of the text of each window slot
  TF-IDF words     linear SVM on word TF-IDF (unigrams and bigrams) of the text of each window slot

The window slots are concatenated in order (-K .. +K); an empty slot (the window runs past the
document's start or end) has all features 0 and, for K > 0, a flag per offset says whether the slot
holds a page. Trained on the annotated trn pages, predicting the annotated tst pages.

PRIVATE TEXT: the public text LMDB is always used; the private one only when --private-text-lmdb is
given. The private text must never be looked at, so this script keeps all text and text-derived
features in memory, and writes and logs only counts and predicted labels. Do not add any output
that carries text (snippets, tokens, vocabularies, per-page features). Develop and debug on the
public text: without --private-text-lmdb, pages whose text is not in the public LMDB are left out
of trn and tst (and empty as neighbours), and the output is named *.public_only.

Writes <work-dir>/k<K>/predictions/text_baselines[.public_only].parquet (key, label and one column
per baseline, for the tst pages with text). Scores are computed by report.py.

Example:
    python -m embedding_probe.text_baselines --context 0                     # public text only
    python -m embedding_probe.text_baselines --context 0 --private-text-lmdb \\
        /run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump.private/lmdb
"""
import argparse
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import polars as pl

from embedding_probe.common import PUBLIC_TEXT_LMDB, WORK_DIR, label_ids, setup_logging, window_ids
from embedding_probe.probe_embeddings import names

logger = logging.getLogger('TEXT BASELINES')

LINE_END_NUMBER = re.compile(r'\d+\s*$')
INDEX_LINE = re.compile(r'^\D{2,},\s*\d')
DOT_LEADER = re.compile(r'\.{3,}|(?:\. ){3,}|…')
LIST_START = re.compile(r'^(?:\d+[.)]|[-–•*])\s')
NUMBER_ONLY = re.compile(r'^[\s.,:;—–-]*\d[\d\s.,:;—–-]*$|^[IVXLCivxlc]+\.?$')  # a page number or roman numeral alone
LAYOUT_FEATURES = ('log_chars', 'log_lines', 'empty', 'mean_line_len', 'std_line_len', 'short_lines',
                   'lines_end_number', 'number_only_lines', 'index_lines', 'dot_leader_lines', 'list_start_lines',
                   'digit_ratio', 'upper_ratio', 'allcaps_lines', 'punct_ratio', 'alpha_ratio', 'words_per_line')


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--context', type=int, default=0, help='Context window K: pages on each side.')
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--public-text-lmdb', type=Path, default=PUBLIC_TEXT_LMDB)
    parser.add_argument('--private-text-lmdb', type=Path, default=None,
                        help='The private text LMDB; read only by this script, never to be looked at.')
    return parser.parse_args(argv)


def layout_features(text: str) -> List[float]:
    """Line-structure statistics of a page text (one OCR line per text line)."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chars = [c for c in text if not c.isspace()]
    n_chars = len(chars)
    if not lines:
        return [0.0, 0.0, 1.0] + [0.0] * (len(LAYOUT_FEATURES) - 3)
    lengths = np.array([len(line) for line in lines], dtype=np.float32)
    letters = [c for c in chars if c.isalpha()]
    n_letters = max(len(letters), 1)
    allcaps = [line for line in lines if sum(c.isalpha() for c in line) >= 3 and line.upper() == line]

    def share(pattern):
        return sum(1 for line in lines if pattern.search(line)) / len(lines)

    return [
        float(np.log1p(n_chars)), float(np.log1p(len(lines))), 0.0,
        float(lengths.mean()), float(lengths.std()), float((lengths < 20).mean()),
        share(LINE_END_NUMBER), share(NUMBER_ONLY), share(INDEX_LINE), share(DOT_LEADER), share(LIST_START),
        sum(c.isdigit() for c in chars) / max(n_chars, 1),
        sum(c.isupper() for c in letters) / n_letters,
        len(allcaps) / len(lines),
        sum(not c.isalnum() for c in chars) / max(n_chars, 1),
        len(letters) / max(n_chars, 1),
        float(np.mean([len(line.split()) for line in lines])),
    ]


def read_texts(keys: List[str], public_lmdb: Path, private_lmdb: Optional[Path]) -> Dict[str, str]:
    """key -> text for the keys found, public LMDB first."""
    texts: Dict[str, str] = {}
    for path in (public_lmdb, private_lmdb):
        if path is None:
            continue
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            for key in sorted(keys):
                if key not in texts:
                    value = txn.get(key.encode('utf-8'))
                    if value is not None:
                        texts[key] = value.decode('utf-8')
        env.close()
    return texts


def slot_rows(slots: List[List[Optional[str]]], row_of: Dict[str, int]) -> List[np.ndarray]:
    """Per window slot: the row of each page's slot text in the unique-text matrix, -1 where empty."""
    return [np.array([row_of.get(k, -1) if k is not None else -1 for k in slot]) for slot in slots]


def window_dense(features: np.ndarray, rows: List[np.ndarray]) -> np.ndarray:
    parts = []
    for r in rows:
        x = np.zeros((len(r), features.shape[1]), dtype=np.float32)
        x[r >= 0] = features[r[r >= 0]]
        parts.append(x)
    return np.hstack(parts)


def window_sparse(features, rows: List[np.ndarray]):
    import scipy.sparse as sp
    parts = []
    for r in rows:
        select = sp.csr_matrix((np.ones(int((r >= 0).sum()), dtype=np.float32),
                                (np.flatnonzero(r >= 0), r[r >= 0])), shape=(len(r), features.shape[0]))
        parts.append(select @ features)
    return sp.hstack(parts, format='csr')


def presence_flags(rows: List[np.ndarray], context: int) -> np.ndarray:
    flags = [(r >= 0).astype(np.float32) for o, r in zip(range(-context, context + 1), rows) if o != 0]
    return np.stack(flags, axis=1) if flags else np.zeros((len(rows[0]), 0), dtype=np.float32)


def main(argv=None):
    args = parse_arguments(argv)
    suffix = '' if args.private_text_lmdb else '.public_only'
    setup_logging('TEXT BASELINES', args.work_dir / 'logs' / f'text_baselines.k{args.context}{suffix}.txt')
    import scipy.sparse as sp
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    pages = pl.read_parquet(args.work_dir / 'pages.parquet')
    slots = window_ids(pages, 'text', args.context)
    keys = sorted({k for slot in slots for k in slot if k is not None})
    started = time.time()
    texts = read_texts(keys, args.public_text_lmdb, args.private_text_lmdb)
    logger.info(f'K={args.context}: text found for {len(texts)} of {len(keys)} window pages, '
                f'{time.time() - started:.0f}s')
    unique = sorted(texts)
    rows = slot_rows(slots, {k: i for i, k in enumerate(unique)})
    keep = rows[args.context] >= 0
    split = pages['split'].to_numpy()
    for s in ('trn', 'tst'):
        logger.info(f'{s}: {int(keep[split == s].sum())} of {int((split == s).sum())} annotated pages with text')
    pages, rows = pages.filter(pl.Series(keep)), [r[keep] for r in rows]
    y = label_ids(pages['label'].to_list())
    trn = (pages['split'] == 'trn').to_numpy()
    tst = ~trn
    flags = presence_flags(rows, args.context)
    predictions = {'key': pages['key'].to_numpy()[tst], 'label': pages['label'].to_numpy()[tst]}

    layout = np.array([layout_features(texts[k]) for k in unique], dtype=np.float32)
    x = np.hstack([window_dense(layout, rows), flags])
    # no early stopping: its stratified validation split fails for classes with a single trn page
    model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, early_stopping=False,
                                           random_state=0).fit(x[trn], y[trn])
    predictions['layout stats'] = names(model.predict(x[tst]))
    logger.info(f'layout stats: {x.shape[1]} features, {time.time() - started:.0f}s')
    del layout, x

    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=200_000,
                            sublinear_tf=True, dtype=np.float32)
    tfidf.fit([texts[unique[i]] for i in rows[args.context][trn]])
    words = tfidf.transform([texts[k] for k in unique])
    del tfidf, texts
    x = sp.hstack([window_sparse(words, rows), sp.csr_matrix(flags)], format='csr')
    model = LinearSVC(C=0.5, class_weight='balanced', max_iter=5000).fit(x[trn], y[trn])
    predictions['TF-IDF words'] = names(model.predict(x[tst]))
    logger.info(f'TF-IDF words: {x.shape[1]} features, {time.time() - started:.0f}s')
    del words, x, model

    out_dir = args.work_dir / f'k{args.context}' / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'text_baselines{suffix}.parquet'
    pl.DataFrame({k: list(v) for k, v in predictions.items()}).write_parquet(out_path)
    logger.info(f'wrote {out_path}')


if __name__ == '__main__':
    main()
