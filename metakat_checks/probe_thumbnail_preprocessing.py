"""Test the naki-tools thumbnail exporter against native DINO vectors.

Run using MetaKat's environment and PYTHONPATH. Only public prepared batches
are read; the supplied resize script is imported, never invoked as a CLI.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_store.reader import TrainingStore


def main():
    from metakat.page_type.engines.core.context.dino import DinoEncoder
    from metakat.page_type.engines.core.context.alto_stats import normalize_dense
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'store', 'resize-script', 'dino-checkpoint', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--batch', action='append', required=True)
    parser.add_argument('--max-size', type=int, default=256)
    parser.add_argument('--quality', type=int)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('thumbnail_exporter', args.resize_script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    quality = exporter.DEFAULT_QUALITY if args.quality is None else args.quality
    if args.max_size < 1 or not 0 <= quality <= 100:
        raise ValueError('Invalid thumbnail size or JPEG quality')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    store = TrainingStore(args.store)
    encoder = DinoEncoder(args.dino_checkpoint, 'cuda', batch_size=32)
    report = {'resize_script': str(args.resize_script.resolve()),
              'resize_script_sha256': hashlib.sha256(args.resize_script.read_bytes()).hexdigest(),
              'opencv': cv2.__version__, 'max_size': args.max_size, 'quality': quality, 'batches': []}
    try:
        for name in args.batch:
            batch = args.root/name
            manifest = json.loads((batch/'manifest.json').read_text())
            native = store.features(store.read_rows([p['row_id'] for p in manifest['pages']]))['image'].astype(np.float32)
            with tempfile.TemporaryDirectory(prefix='metakat-thumbnails-') as directory:
                paths = []
                for index, page in enumerate(manifest['pages']):
                    image = cv2.imread(str(batch/page['image']), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError('Image decode failed: '+page['image'])
                    image = exporter.resize_to_max_size(image, args.max_size)
                    target = Path(directory)/f'{index:06d}.jpg'
                    if not cv2.imwrite(str(target), image, exporter.imwrite_params(target, quality)):
                        raise IOError('JPEG encoding failed')
                    paths.append(str(target))
                raw, present = encoder.encode(paths)
            vectors = normalize_dense(raw, present)
            cosine = (vectors*native).sum(-1)/(np.linalg.norm(vectors, axis=-1)*np.linalg.norm(native, axis=-1))
            np.savez_compressed(args.output/(name+'.image.npz'), image=vectors)
            entry = {'batch': name, 'pages': len(vectors), 'mean_cosine': float(cosine.mean()),
                     'min_cosine': float(cosine.min()), 'max_abs': float(np.max(np.abs(vectors-native)))}
            report['batches'].append(entry)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(entry), flush=True)
    finally:
        store.close()


if __name__ == '__main__':
    main()
