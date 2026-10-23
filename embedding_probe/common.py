"""Shared definitions: the CSV format, the label space, default locations and the evaluation metrics."""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

SSD_DATA = Path('/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data')
CSV_DIR = Path('/mnt/matylda0/ihradis/digiknihovna.data/page_types_metadata_for_training')
# the copy of digiknihovna.data/{public,private}/{text,image}/<model>/ on kolosus (matylda0 is slow to read at random)
EMBEDDINGS_ROOT = Path('/mnt/kolosus/data/digiknihovna.data')
WORK_DIR = SSD_DATA / 'embedding_probe'
PUBLIC_TEXT_LMDB = SSD_DATA / '2026-09-16.db_text_dump' / 'lmdb'
# the private text LMDB is SSD_DATA / '2026-09-16.db_text_dump.private' / 'lmdb'; it is passed to the scripts
# explicitly and never read other than by them (see CLAUDE.md)

CSV_COLUMNS = ('library', 'root', 'parent', 'doc', 'page', 'db_type', 'annotated_type', 'placement', 'order',
               'pagenum', 'date', 'access', 'path')
ANNOTATED_CSV = {'trn': 'annotated.trn.256.final.csv', 'tst': 'annotated.tst.256.final.csv'}
CONTEXT_CSV = {'trn': 'db.256.final.pruned.trn_docs.csv', 'tst': 'db.256.final.pruned.tst_docs.csv'}

# MetaKat's 38 page type classes (metakat/page_type/datasets_from_mods/mods_helper.py) plus Colophon
PAGE_TYPES = (
    'Abstract', 'Advertisement', 'Appendix', 'BackCover', 'BackEndPaper', 'BackEndSheet', 'Bibliography', 'Blank',
    'CalibrationTable', 'Colophon', 'Cover', 'CustomInclude', 'Dedication', 'Edge', 'Errata', 'FlyLeaf',
    'FragmentsOfBookbinding', 'FrontCover', 'FrontEndPaper', 'FrontEndSheet', 'FrontJacket', 'Frontispiece',
    'Illustration', 'Impressum', 'Imprimatur', 'Index', 'Jacket', 'ListOfIllustrations', 'ListOfMaps',
    'ListOfTables', 'Map', 'NormalPage', 'Obituary', 'Preface', 'SheetMusic', 'Spine', 'Table', 'TableOfContents',
    'TitlePage')
LABEL2ID = {name: i for i, name in enumerate(PAGE_TYPES)}

# embedding folders (digiknihovna.data/{public,private}/<kind>/<model>/) by short name
MODELS = {
    'mmBERT': ('text', 'mmBERT-base'),
    'Qwen3': ('text', 'Qwen3-Embedding-0.6B'),
    'DINOv2': ('image', 'vitb14_dinov2_2026-09-20_public_private_2'),
}
IMAGE_ROOTS = {'public.256': 'public', 'private.256': 'private'}


def setup_logging(name: str, log_file: Path = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, format=f'{name} - %(asctime)s - %(message)s', handlers=handlers)
    logging.getLogger(name).info(' '.join(sys.argv))


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, has_header=False, new_columns=list(CSV_COLUMNS), infer_schema=False)


def text_key(library: str, page: str) -> str:
    """Key of the page-text and text-embedding LMDBs."""
    return f'{library}_{page}'


def image_key(path: str):
    """CSV image path -> (public|private, key of the image-embedding LMDB)."""
    root, key = path.split('/', 1)
    return IMAGE_ROOTS[root], key


def label_ids(labels: Sequence[str]) -> np.ndarray:
    """Class ids; -1 for anything outside the label space (e.g. an empty or extra DB page type)."""
    return np.array([LABEL2ID.get(label, -1) if label is not None else -1 for label in labels], dtype=np.int64)


def evaluate(gt: Sequence[str], pred: Sequence[str]) -> Dict:
    """The metrics of the README: accuracy, weighted precision/recall/F1 and per-class values over all class ids.

    A prediction outside the label space counts as wrong.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    gt, pred = label_ids(gt), label_ids(pred)
    assert (gt >= 0).all(), 'a ground-truth label outside the label space'
    labels = list(range(len(LABEL2ID)))
    w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(
        gt, pred, average='weighted', labels=labels, zero_division=0)
    precision, recall, fscore, support = precision_recall_fscore_support(
        gt, pred, average=None, labels=labels, zero_division=0)
    return {
        'n': int(len(gt)),
        'accuracy': float(accuracy_score(gt, pred)),
        'weighted_precision': float(w_precision),
        'weighted_recall': float(w_recall),
        'weighted_fscore': float(w_fscore),
        'per_class': {PAGE_TYPES[i]: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                      'fscore': float(fscore[i]), 'support': int(support[i])}
                      for i in labels if support[i] > 0},
    }


def window_ids(pages: pl.DataFrame, kind: str, context: int) -> List[List[Optional[str]]]:
    """Per offset -context .. +context: the vector id of each page's window slot (None where empty).

    The vector id is the text key {library}_{page uuid} for text models and the CSV image path for
    image models (the image LMDB key is the path without its root, which selects the LMDB).
    """
    max_context = pages['window_page'].list.len().max() // 2
    assert context <= max_context, f'pages.parquet holds windows up to {max_context} only'
    column = 'window_page' if kind == 'text' else 'window_path'
    result = []
    for offset in range(-context, context + 1):
        slot = pages[column].list.get(max_context + offset).to_list()
        if kind == 'text':
            slot = [None if p is None else text_key(lib, p) for lib, p in zip(pages['library'].to_list(), slot)]
        result.append(slot)
    return result


def window_completeness(pages: pl.DataFrame, context: int) -> np.ndarray:
    """full | left cut | right cut | both cut: whether the window of K = context reaches past the document."""
    if context == 0:
        return np.full(len(pages), 'full', dtype=object)
    max_context = pages['window_page'].list.len().max() // 2
    left = pages['window_page'].list.get(max_context - context).is_null().to_numpy()
    right = pages['window_page'].list.get(max_context + context).is_null().to_numpy()
    return np.select([left & right, left, right], ['both cut', 'left cut', 'right cut'], 'full').astype(object)
