"""Extracts the embedding vectors of all pages the experiments can use, onto the work directory:
the annotated pages and their context windows up to the --max-context of pages.parquet.

Reads the embedding LMDBs in <embeddings-root>/{public,private}/{text,image}/<model>/lmdb (read
only) and writes <work-dir>/vectors/<name>.npz:

  ids       the vector ids, sorted: {library}_{page uuid} for text models, the CSV image path for images
  vectors   float16 (n, dim), the raw vectors as stored (not normalized)
  source    1 = found in the public LMDB, 2 = private
  meta      the model's meta.json

Text vectors are keyed {library}_{page uuid}; a page is in the public or the private text LMDB.
Image vectors are keyed by the CSV image path without its first component, whose public.256 /
private.256 selects the LMDB. The ids are looked up in sorted order by parallel threads, each with
its own read transaction (py-lmdb releases the GIL during a lookup).

Example:
    python -m embedding_probe.extract_vectors --models mmBERT Qwen3 DINOv2
"""
import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import polars as pl

from embedding_probe.common import EMBEDDINGS_ROOT, MODELS, WORK_DIR, image_key, setup_logging, window_ids

logger = logging.getLogger('EXTRACT VECTORS')
SOURCES = {'public': 1, 'private': 2}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    parser.add_argument('--embeddings-root', type=Path, default=EMBEDDINGS_ROOT)
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--threads', type=int, default=16, help='Parallel LMDB readers.')
    return parser.parse_args(argv)


def needed_ids(pages: pl.DataFrame, kind: str) -> List[str]:
    max_context = pages['window_page'].list.len().max() // 2
    ids = {i for slot in window_ids(pages, kind, max_context) for i in slot if i is not None}
    return sorted(ids)


def lmdb_requests(vector_id: str, kind: str) -> Tuple[Tuple[str, ...], str]:
    """The stores to try and the LMDB key of a vector id."""
    if kind == 'text':
        return ('public', 'private'), vector_id
    store, key = image_key(vector_id)
    return (store,), key


def read_vectors(envs: Dict[str, lmdb.Environment], ids: List[str], kind: str, dtype: str,
                 threads: int) -> Dict[str, Tuple[int, np.ndarray]]:
    """vector id -> (source, vector) for the ids found."""
    found: Dict[str, Tuple[int, np.ndarray]] = {}
    lock = threading.Lock()
    started = time.time()
    for store in ('public', 'private'):
        todo = [(i, lmdb_requests(i, kind)[1]) for i in ids
                if store in lmdb_requests(i, kind)[0] and i not in found]
        todo.sort(key=lambda item: item[1])
        if not todo:
            continue
        env = envs[store]
        done = [0]

        def read(chunk):
            result = {}
            with env.begin() as txn:
                for vector_id, key in chunk:
                    value = txn.get(key.encode('utf-8'))
                    if value is not None:
                        result[vector_id] = (SOURCES[store], np.frombuffer(value, dtype=dtype).copy())
            with lock:
                done[0] += len(chunk)
                logger.info(f'{store}: {done[0]} of {len(todo)} looked up, {time.time() - started:.0f}s')
            return result

        size = max(1, min(2000, -(-len(todo) // threads)))
        with ThreadPoolExecutor(threads) as pool:
            for result in pool.map(read, [todo[i:i + size] for i in range(0, len(todo), size)]):
                found.update(result)
    return found


def extract_model(pages: pl.DataFrame, embeddings_root: Path, name: str, out_path: Path, threads: int):
    kind, folder = MODELS[name]
    dirs = {store: embeddings_root / store / kind / folder for store in SOURCES}
    meta = json.loads((dirs['public'] / 'meta.json').read_text(encoding='utf-8'))
    envs = {store: lmdb.open(str(d / 'lmdb'), readonly=True, lock=False, readahead=False,
                             max_readers=threads + 8) for store, d in dirs.items()}
    ids = needed_ids(pages, kind)
    logger.info(f'{name}: {len(ids)} vectors needed')
    started = time.time()
    found = read_vectors(envs, ids, kind, meta['dtype'], threads)
    for env in envs.values():
        env.close()
    missing = [i for i in ids if i not in found]
    logger.info(f'{name}: {len(found)} found in {time.time() - started:.0f}s, {len(missing)} missing'
                + (f', e.g. {missing[:3]}' if missing else ''))
    present = [i for i in ids if i in found]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ids=np.array(present, dtype=str), meta=json.dumps(meta),
             vectors=np.stack([found[i][1] for i in present]).astype(np.float16),
             source=np.array([found[i][0] for i in present], dtype=np.uint8))
    logger.info(f'wrote {out_path}')


class Vectors:
    """The extracted vectors of one model, looked up by vector id."""

    def __init__(self, work_dir: Path, name: str):
        data = np.load(work_dir / 'vectors' / f'{name}.npz')
        self.name, self.kind = name, MODELS[name][0]
        self.meta = json.loads(str(data['meta']))
        self.vectors, self.source = data['vectors'], data['source']
        self.index = {i: n for n, i in enumerate(data['ids'].tolist())}

    def window(self, ids: List[Optional[str]]):
        """(n, dim) float32 vectors of the ids, zeros where None or not found, and the found mask."""
        rows = np.array([self.index.get(i, -1) if i is not None else -1 for i in ids])
        present = rows >= 0
        out = np.zeros((len(ids), self.vectors.shape[1]), dtype=np.float32)
        out[present] = self.vectors[rows[present]]
        return out, present


def main(argv=None):
    args = parse_arguments(argv)
    setup_logging('EXTRACT VECTORS', args.work_dir / 'logs' / 'extract_vectors.txt')
    pages = pl.read_parquet(args.work_dir / 'pages.parquet')
    for name in args.models:
        extract_model(pages, args.embeddings_root, name, args.work_dir / 'vectors' / f'{name}.npz', args.threads)


if __name__ == '__main__':
    main()
