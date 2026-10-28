"""Compute page image embeddings with a LightlyTrain-finetuned DINOv2 ViT-B/14 (with registers).

Input: an image directory with one subdirectory per library, which holds one
subdirectory per document (<document-uuid>.images) with the page images as
<page-uuid>.jpg; the key of each page is its path relative to the image
directory, "<library>/<document>/<page-uuid>.jpg". (The page-text LMDB keys
pages as "{library}_{page-uuid}" instead, which is not unique for images: a page
shared by two documents is stored under both.) Output: <output-root>/<model-name>/
with the same layout as the text embeddings (see
embeddings/text/inference.py):

  lmdb/          the embeddings, raw little-endian vectors
                 (``np.frombuffer(value, dtype=meta['dtype'])``)
  meta.json      checkpoint (path and sha256), pooling, image preprocessing,
                 dtype, source directory; a resumed run with different settings is refused
  file_list.tsv  the sorted key -> image path list, made on the first run and reused
                 on resume, so the processing order stays fixed even if files are added
  progress.tsv   one line per committed chunk; the last line's key is where a resumed
                 run continues
  failed.tsv     images that could not be read (key, path, error); not in the LMDB
  log.txt        the run log (also printed to stdout)
  run.lock       held while a run is writing; a second concurrent run refuses to start

Preprocessing is that of MetaKat's DinoEncoder (metakat/page_type/engines/core/context/dino.py), which the
training vectors were computed with: the page is shrunk to the 256 px thumbnail the original dump was made from
(longest edge 512 px, JPEG quality 90, then 256 px, JPEG quality 90, cv2.INTER_AREA, see thumbnail.py), then
resized to 224x224 with cv2.INTER_AREA and ImageNet-normalized. A page that already is a thumbnail of at most
256 px is used as it is, so the input directory may hold either those thumbnails or full-resolution scans, which
give the same vectors. JPEG 2000 is decoded at a reduced resolution level that stays at least 1024 px on the
longest edge.

The embedding is the CLS token after the final LayerNorm (768-d); "cls_mean"
concatenates it with the mean patch token (1536-d, the DINOv2 linear-probe
features) and "mean" is the mean patch token alone.

Example:
    python -m embeddings.image.dinov2.infer_dinov2 \\
        --image-dir /data/page_images \\
        --checkpoint /data/.../vitb14_dinov2_2026-09-18_public/exported_models/exported_last.pt \\
        --output-root /data/2026-09-16.db_image_dump/embeddings
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from embeddings.gpu_bootstrap import add_gpu_arguments, bootstrap_single_gpu


# CUDA_VISIBLE_DEVICES must be finalized before importing PyTorch or Transformers.
if __name__ == '__main__':
    bootstrap_single_gpu(sys.argv[1:])


import logging

import cv2
import lmdb
import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from embeddings.image.dinov2.dinov2_checkpoint import load_dinov2_with_registers
from embeddings.image.dinov2.thumbnail import THUMBNAIL_SIZES, original_thumbnail
from embeddings.run_state import (
    StopRequest,
    acquire_run_lock,
    append_progress,
    check_or_write_meta,
    load_progress,
    setup_logging,
)

logger = logging.getLogger(__name__)

PROGRESS_HEADER = ('timestamp', 'chunk_index', 'first_key', 'last_key', 'chunk_size', 'failed', 'total_done',
                   'images_per_s')
FIXED_META_FIELDS = ('checkpoint_sha256', 'architecture', 'pooling', 'image_size', 'thumbnail', 'resize', 'dtype', 'dim',
                     'source_images')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
Image.MAX_IMAGE_PIXELS = None  # full-resolution scans and long strips exceed PIL's decompression-bomb limit


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Compute DINOv2 page image embeddings for a directory of images.')
    parser.add_argument('--image-dir', type=Path, required=True,
                        help='Directory of <library>/<document>/<page-uuid>.jpg page images.')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='LightlyTrain exported_models/exported_last.pt (original DINOv2 state dict).')
    parser.add_argument('--architecture', default='vitb14', choices=['vits14', 'vitb14', 'vitl14'])
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Embeddings root; output goes to <output-root>/<model-name>/.')
    parser.add_argument('--model-name', default=None,
                        help='Output subdirectory name (default: the LightlyTrain experiment directory name).')
    parser.add_argument('--extensions', default='.jpg,.jpeg',
                        help='Comma-separated image file extensions to include (case-insensitive).')

    parser.add_argument('--image-size', type=int, default=224, help='Model input size (a multiple of 14).')
    parser.add_argument('--pooling', choices=['cls', 'cls_mean', 'mean'], default='cls')

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=min(10, os.cpu_count() or 1),
                        help='DataLoader processes decoding images.')
    parser.add_argument('--chunk-size', type=int, default=8192, help='Images per LMDB commit / progress line.')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16',
                        help='Storage dtype of the embedding vectors.')
    parser.add_argument('--compute-dtype', choices=['bfloat16', 'float16', 'float32'], default='float16',
                        help='Autocast dtype of the forward pass. float16 matches float32 to cosine >= 0.9997 at '
                             'the same speed as bfloat16, whose CLS tokens drift to ~0.988 on some pages; '
                             'float32 is ~3x slower.')
    parser.add_argument('--attn-implementation', choices=['sdpa', 'eager'], default='sdpa')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after this many images in this run (for testing); resumable as usual.')
    parser.add_argument('--map-size-gb', type=int, default=300,
                        help='Output LMDB map_size in GB (address-space reservation, not preallocation).')
    parser.add_argument('--logging-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    add_gpu_arguments(parser)
    return parser.parse_args(argv)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _is_image(entry, extensions) -> bool:
    return os.path.splitext(entry.name)[1].lower() in extensions and (entry.is_file() or entry.is_symlink())


def build_file_list(image_dir: Path, extensions, list_path: Path) -> int:
    """Lists <image-dir>/<library>/<document>/<page-uuid>.<ext> into a key-sorted "key<TAB>relative path" file.

    The key is the relative path itself. Images lying directly in a library directory are not pages
    of a document and are skipped (with a warning).
    """
    entries = []
    for library in sorted(e.name for e in os.scandir(image_dir) if e.is_dir()):
        count = documents = skipped = 0
        with os.scandir(image_dir / library) as library_it:
            for document in library_it:
                if not document.is_dir():
                    skipped += _is_image(document, extensions)
                    continue
                documents += 1
                with os.scandir(document.path) as it:
                    for entry in it:
                        if _is_image(entry, extensions):
                            path = f'{library}/{document.name}/{entry.name}'
                            entries.append((path, path))
                            count += 1
        logger.info(f'Listed {count} images in {documents} documents of {library}')
        if skipped:
            logger.warning(f'Skipped {skipped} images directly in {library}/, outside any document directory')
    entries.sort()
    temp_path = list_path.with_name(list_path.name + '.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        for key, path in entries:
            f.write(f'{key}\t{path}\n')
    temp_path.replace(list_path)
    return len(entries)


def read_file_list(list_path: Path, after_key, limit):
    """Returns the (keys, paths) after after_key as fixed-width byte arrays (cheap to share with workers)."""
    keys, paths = [], []
    after = after_key.decode('utf-8') if after_key is not None else None
    with list_path.open(encoding='utf-8') as f:
        for line in f:
            key, path = line.rstrip('\n').split('\t')
            if after is not None and key <= after:
                continue
            keys.append(key)
            paths.append(path)
            if limit is not None and len(keys) >= limit:
                break
    return np.array(keys, dtype=np.bytes_), np.array(paths, dtype=np.bytes_)


def load_image(path: str, image_size: int) -> np.ndarray:
    """Decodes one page, makes its original 256 px thumbnail and resizes that to image_size x image_size."""
    with Image.open(path) as image:
        if image.format == 'JPEG2000':
            reduce = 0
            while max(image.size) >> (reduce + 1) >= 2 * THUMBNAIL_SIZES[0] and reduce < 5:
                reduce += 1
            image.reduce = reduce
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return cv2.resize(original_thumbnail(rgb), (image_size, image_size), interpolation=cv2.INTER_AREA)


class PageImages(Dataset):
    def __init__(self, image_dir: Path, paths: np.ndarray, image_size: int):
        self.image_dir = str(image_dir)
        self.paths = paths
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.image_dir, self.paths[index].decode('utf-8'))
        try:
            return index, torch.from_numpy(load_image(path, self.image_size)), ''
        except Exception as exc:  # unreadable or truncated image: reported, not fatal
            return index, torch.zeros((self.image_size, self.image_size, 3), dtype=torch.uint8), \
                f'{type(exc).__name__}: {exc}'


def collate(items):
    indices, images, errors = zip(*items)
    return torch.tensor(indices), torch.stack(images), list(errors)


@torch.inference_mode()
def embed(model, images: torch.Tensor, device, compute_dtype, pooling: str, num_registers: int) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    pixels = images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255).sub_(mean).div_(std)
    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=compute_dtype != torch.float32):
        tokens = model(pixel_values=pixels).last_hidden_state  # after the final LayerNorm
    tokens = tokens.float()
    cls = tokens[:, 0]
    patch_mean = tokens[:, 1 + num_registers:].mean(dim=1)
    pooled = {'cls': cls, 'mean': patch_mean, 'cls_mean': torch.cat([cls, patch_mean], dim=1)}[pooling]
    return pooled.cpu().numpy()


def main(argv=None):
    args = parse_arguments(argv)
    if args.image_size % 14:
        raise SystemExit('--image-size must be a multiple of the 14-pixel patch size.')
    extensions = {e.strip().lower() if e.strip().startswith('.') else '.' + e.strip().lower()
                  for e in args.extensions.split(',') if e.strip()}

    model_name = args.model_name or args.checkpoint.resolve().parent.parent.name
    output_dir = args.output_root / model_name
    lmdb_dir = output_dir / 'lmdb'
    progress_path = output_dir / 'progress.tsv'
    list_path = output_dir / 'file_list.tsv'
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    lock_file = acquire_run_lock(output_dir)  # noqa: F841 -- held until the process exits

    setup_logging(output_dir / 'log.txt', args.logging_level)
    logger.info(' '.join(sys.argv))
    logger.info(f'torch {torch.__version__}, transformers {transformers.__version__}, '
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_dtype = getattr(torch, args.compute_dtype)
    model = load_dinov2_with_registers(args.checkpoint, args.architecture, args.attn_implementation)
    model.to(device).eval()
    num_registers = model.config.num_register_tokens
    dim = model.config.hidden_size * (2 if args.pooling == 'cls_mean' else 1)

    if list_path.exists():
        with list_path.open(encoding='utf-8') as f:
            total = sum(1 for _ in f)
        logger.info(f'Using the existing file list {list_path} ({total} images).')
    else:
        start = time.time()
        total = build_file_list(args.image_dir, extensions, list_path)
        logger.info(f'Listed {total} images in {time.time() - start:.0f}s into {list_path}.')

    meta = {
        'model': 'DINOv2 with registers (LightlyTrain export)',
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': sha256_of(args.checkpoint),
        'architecture': args.architecture,
        'pooling': {'cls': 'cls_token', 'mean': 'mean_patch_tokens',
                    'cls_mean': 'cls_token+mean_patch_tokens'}[args.pooling],
        'normalized': False,
        'image_size': args.image_size,
        'thumbnail': 'full resolution -> 512 px JPEG Q90 -> 256 px JPEG Q90 (cv2.INTER_AREA), see thumbnail.py',
        'resize': 'thumbnail to image_size x image_size, cv2.INTER_AREA, aspect ratio not kept',
        'image_normalization': {'mean': IMAGENET_MEAN, 'std': IMAGENET_STD},
        'dtype': args.dtype,
        'compute_dtype': args.compute_dtype,
        'dim': dim,
        'source_images': str(args.image_dir.resolve()),
        'source_entries': total,
        'key_format': '<library>/<document>/<page-uuid>.<ext>: image path relative to source_images',
        'value_format': f'raw {args.dtype} bytes, np.frombuffer(value, dtype="{args.dtype}")',
    }
    check_or_write_meta(output_dir / 'meta.json', meta, FIXED_META_FIELDS)

    last_key, total_done, chunk_index = load_progress(progress_path, PROGRESS_HEADER)
    keys, paths = read_file_list(list_path, last_key, args.limit)
    if last_key is not None:
        logger.info(f'Resuming after key {last_key.decode()} ({total_done}/{total} done, chunk {chunk_index}).')
    logger.info(f'{len(keys)} images to process in this run.')
    if len(keys) == 0:
        logger.info('Nothing to do.')
        return

    loader = DataLoader(PageImages(args.image_dir, paths, args.image_size), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate, pin_memory=device.type == 'cuda',
                        persistent_workers=False, prefetch_factor=4 if args.num_workers else None)
    output_env = lmdb.open(str(lmdb_dir), map_size=args.map_size_gb * 1024 ** 3)
    stop = StopRequest()
    store_dtype = np.dtype(args.dtype)
    run_done = 0
    run_start = time.time()
    chunk_start = time.time()
    buffer_index, buffer_vectors, failures = [], [], []

    def commit():
        nonlocal chunk_index, total_done, run_done, chunk_start, buffer_index, buffer_vectors, failures
        indices = np.concatenate(buffer_index)
        vectors = np.concatenate(buffer_vectors).astype(store_dtype)
        failed = {i for i, _ in failures}
        ok = np.array([i not in failed for i in indices])
        bad_rows = np.flatnonzero(ok & ~np.isfinite(vectors).all(axis=1))
        if bad_rows.size:
            raise RuntimeError(f'{bad_rows.size} non-finite embeddings in chunk {chunk_index}, e.g. key '
                               f'{keys[indices[bad_rows[0]]].decode()}; nothing of this chunk was stored.')
        with output_env.begin(write=True) as txn:
            for i, vector in zip(indices[ok], vectors[ok]):
                txn.put(keys[i], vector.tobytes())
        if failures:
            with (output_dir / 'failed.tsv').open('a', encoding='utf-8') as f:
                for i, error in failures:
                    f.write(f'{keys[i].decode()}\t{paths[i].decode()}\t{error.replace(chr(9), " ")}\n')
        total_done += len(indices)
        run_done += len(indices)
        rate = len(indices) / max(time.time() - chunk_start, 1e-9)
        append_progress(progress_path, (
            time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), chunk_index, keys[indices[0]].decode(),
            keys[indices[-1]].decode(), len(indices), len(failures), total_done, f'{rate:.1f}'), PROGRESS_HEADER)
        run_rate = run_done / max(time.time() - run_start, 1e-9)
        eta_h = (total - total_done) / max(run_rate, 1e-9) / 3600
        logger.info(f'chunk {chunk_index}: {len(indices)} images ({len(failures)} failed), {rate:.1f} images/s | '
                    f'total {total_done}/{total} ({100 * total_done / total:.2f}%), run avg {run_rate:.1f} '
                    f'images/s, ETA {eta_h:.1f} h')
        chunk_index += 1
        chunk_start = time.time()
        buffer_index, buffer_vectors, failures = [], [], []

    try:
        buffered = 0
        for indices, images, errors in loader:
            vectors = embed(model, images, device, compute_dtype, args.pooling, num_registers)
            buffer_index.append(indices.numpy())
            buffer_vectors.append(vectors)
            failures.extend((int(i), e) for i, e in zip(indices, errors) if e)
            buffered += len(indices)
            if buffered >= args.chunk_size:
                commit()
                buffered = 0
                if stop.requested:
                    logger.info('Stopped on request; rerun the same command to resume.')
                    break
        else:
            if buffered:
                commit()
        output_env.sync()
    finally:
        output_env.close()

    logger.info(f'Run finished: {run_done} images this run, {total_done}/{total} total.')


if __name__ == '__main__':
    main()
