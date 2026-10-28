"""Compares page-text normalizations of mmBERT by the linear probe of probe_embeddings.py.

The variants are the stored mmBERT vectors of the text dump as it is ("old"), and vectors computed by
embeddings/text/mmbert/infer_mmbert.py with --normalization ocr_lines and ocr_words. Only annotated pages
with public text are used: the text of a private page is never read, and public text may be embedded by a
script. DINOv2 is the stored image embedding, so the text+image probe is compared too.

Two steps, with the inference between them:

  prepare   writes <work-dir>/pages.parquet (annotated pages with public text), vectors/mmBERT_old.npz,
            vectors/DINOv2.npz and text_source/ (an LMDB of the public text of every vector id the windows
            of those pages need), the input of the two inference runs
  probe     reads the inference outputs <work-dir>/embeddings/mmBERT.{ocr_lines,ocr_words}/lmdb, makes
            vectors/mmBERT_{lines,words}.npz, runs the probes for K = --contexts and prints the scores

Example:
    python -m embedding_probe.compare_text_normalization prepare
    for n in ocr_lines ocr_words; do
        python -m embeddings.text.mmbert.infer_mmbert --source-lmdb $W/text_source --output-root $W/embeddings \\
            --model-name mmBERT.$n --normalization $n --model <mmBERT directory>
    done
    python -m embedding_probe.compare_text_normalization probe
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import lmdb
import numpy as np
import polars as pl

from embedding_probe import common, probe_embeddings
from embedding_probe.common import PAGE_TYPES, WORK_DIR, evaluate, text_key

logger = logging.getLogger('COMPARE TEXT NORMALIZATION')
TEXT_LMDB = Path('/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump/lmdb')
VARIANTS = {'mmBERT_old': 'stored (dump text)', 'mmBERT_lines': 'ocr_lines', 'mmBERT_words': 'ocr_words'}
INFERENCE_NAMES = {'mmBERT_lines': 'mmBERT.ocr_lines', 'mmBERT_words': 'mmBERT.ocr_words'}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('step', choices=('prepare', 'probe'))
    parser.add_argument('--source-work-dir', type=Path, default=WORK_DIR, help='The probe work directory.')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--contexts', type=int, nargs='+', default=[0, 2])
    return parser.parse_args(argv)


def prepare(args):
    source = np.load(args.source_work_dir / 'vectors' / 'mmBERT.npz')
    public = source['source'] == 1
    ids = source['ids'][public]
    pages = pl.read_parquet(args.source_work_dir / 'pages.parquet')
    in_public = np.isin([text_key(lib, page) for lib, page in pages.select('library', 'page').iter_rows()], ids)
    pages = pages.filter(pl.Series(in_public))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / 'vectors').mkdir(exist_ok=True)
    pages.write_parquet(args.work_dir / 'pages.parquet')
    np.savez(args.work_dir / 'vectors' / 'mmBERT_old.npz', ids=ids, vectors=source['vectors'][public],
             source=source['source'][public], meta=source['meta'])
    shutil.copy(args.source_work_dir / 'vectors' / 'DINOv2.npz', args.work_dir / 'vectors' / 'DINOv2.npz')
    with lmdb.open(str(TEXT_LMDB), readonly=True, lock=False) as text, \
            lmdb.open(str(args.work_dir / 'text_source'), map_size=2 ** 30) as out:
        with text.begin() as read, out.begin(write=True) as write:
            for vector_id in sorted(ids.tolist()):
                write.put(vector_id.encode(), bytes(read.get(vector_id.encode())))
    split = pages.group_by('split').len().sort('split').rows()
    logger.info(f'{len(pages)} annotated pages with public text {split}; {len(ids)} text vectors/texts')


def to_vectors(args):
    for name, folder in INFERENCE_NAMES.items():
        directory = args.work_dir / 'embeddings' / folder
        meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
        with lmdb.open(str(directory / 'lmdb'), readonly=True, lock=False) as env, env.begin() as txn:
            items = sorted((bytes(k).decode(), bytes(v)) for k, v in txn.cursor())
        vectors = np.stack([np.frombuffer(v, dtype=meta['dtype']) for _, v in items])
        np.savez(args.work_dir / 'vectors' / f'{name}.npz', ids=np.array([k for k, _ in items]), vectors=vectors,
                 source=np.ones(len(items), dtype=np.int8), meta=json.dumps(meta))


def probe(args):
    to_vectors(args)
    common.MODELS.update({name: ('text', name) for name in VARIANTS})
    rows = {}
    for context in args.contexts:
        experiments = [name for name in VARIANTS] + [f'{name}+DINOv2' for name in VARIANTS]
        only = experiments + [f'{name} kNN' for name in VARIANTS]
        probe_embeddings.main(['--work-dir', str(args.work_dir), '--context', str(context), '--models',
                               *VARIANTS, 'DINOv2', '--only', *only])
        predictions = pl.read_parquet(args.work_dir / f'k{context}' / 'predictions' / 'embeddings.parquet')
        for name in only:
            rows[(context, name)] = evaluate(predictions['label'].to_list(), predictions[name].to_list())
    (args.work_dir / 'scores.json').write_text(json.dumps({f'K{k} {n}': v for (k, n), v in rows.items()}, indent=1))
    print(f'{"K":>2} {"experiment":<22} {"n":>4} {"accuracy":>9} {"weighted F1":>12}')
    for (context, name), score in rows.items():
        print(f'{context:>2} {name:<22} {score["n"]:>4} {score["accuracy"]:>9.4f} {score["weighted_fscore"]:>12.4f}')


def main(argv=None):
    args = parse_arguments(argv)
    common.setup_logging('COMPARE TEXT NORMALIZATION')
    {'prepare': prepare, 'probe': probe}[args.step](args)


if __name__ == '__main__':
    main()
