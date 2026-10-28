"""Resumable page text embedding inference over a page-text LMDB, shared by the infer_<model>.py scripts.

Reads a source LMDB of ``{library}_{page_id}`` -> UTF-8 page text and writes a
parallel LMDB with the same keys whose values are the page embeddings. The
output directory (<output-root>/<model-name>/) holds:

  lmdb/          the embeddings, raw little-endian vectors
                 (``np.frombuffer(value, dtype=meta['dtype'])``)
  meta.json      model, commit hash, pooling, dtype, max_length, source path;
                 a resumed run with different settings is refused
  progress.tsv   one line per committed chunk; the last line's key is where a
                 resumed run continues
  log.txt        the run log (also printed to stdout)
  run.lock       held while a run is writing; a second concurrent run refuses to start

Source keys are processed in LMDB key order, one chunk at a time: the chunk's
embeddings are committed to LMDB first, then its progress line is appended and
fsynced. A run killed between the two recomputes that chunk on resume and
overwrites identical values, so the output never holds a key the progress file
does not cover beyond the chunk in flight.

The page text is first normalized (--normalization, see text/normalization.py: by default only the words, with no
layout). Pages whose text is empty after that are all encoded as "", so they
share one embedding. Batches are right-padded; each model's EncoderSpec
decides how the last hidden state is pooled into one vector, computed in
float32 from an autocast (bf16 by default) forward pass. Vectors are stored
unnormalized.

--n-gpu N (default 1) keeps one model replica per GPU, each fed whole chunks by
its own thread. Chunks still commit one at a time in key order, so the output
layout and resume are the same for any N, and a run may resume with another N.

Scripts importing this module must call bootstrap_gpus(..., multi_gpu=True)
before it, since it imports torch.
"""

import argparse
import collections
import copy
import functools
import importlib.util
import json
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple

import lmdb
import numpy as np
import torch
import transformers

from embeddings.gpu_bootstrap import add_gpu_arguments
from embeddings.run_state import (
    StopRequest,
    acquire_run_lock,
    append_progress,
    check_or_write_meta,
    load_progress,
    setup_logging,
)
from embeddings.text.normalization import MAX_CHARS, NORMALIZATIONS

logger = logging.getLogger(__name__)

PROGRESS_HEADER = ('timestamp', 'chunk_index', 'first_key', 'last_key', 'chunk_size',
                   'empty', 'truncated', 'total_done', 'docs_per_s')
FIXED_META_FIELDS = ('model', 'model_commit_hash', 'pooling', 'dtype', 'dim', 'max_length', 'empty_text',
                     'source_lmdb', 'normalization', 'max_chars')


@dataclass
class EncoderSpec:
    """What differs between embedding models; everything else is shared."""

    description: str
    default_model: str
    pooling: str
    # (model id, attn_implementation, compute dtype) -> (tokenizer, model); moved to the device afterwards
    load: Callable[[str, str, torch.dtype], tuple]
    # (last hidden state [B, T, D], right-padded attention mask [B, T]) -> float32 [B, D]
    pool: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    default_max_length: int = 8192
    default_max_tokens_per_batch: int = 65536
    # None: flash_attention_2 if flash-attn is installed, else sdpa
    default_attn_implementation: Optional[str] = None
    max_length_help: str = ''
    # adds model-specific arguments to the parser
    add_arguments: Callable[[argparse.ArgumentParser], None] = lambda parser: None
    # (args) -> text transform applied after empty-page normalization, e.g. an instruction prefix
    make_text_transform: Callable[[argparse.Namespace], Optional[Callable[[str], str]]] = lambda args: None
    # (args) -> meta fields that must also match on resume
    extra_meta: Callable[[argparse.Namespace], dict] = lambda args: {}
    extra_fixed_fields: Tuple[str, ...] = field(default_factory=tuple)


class Chunk(NamedTuple):
    keys: List[bytes]
    input_ids: List[List[int]]
    empty: int
    truncated: int


def parse_arguments(spec: EncoderSpec, argv=None):
    parser = argparse.ArgumentParser(description=spec.description)

    parser.add_argument('--source-lmdb', type=Path, required=True,
                        help='Source LMDB directory of "{library}_{page_id}" -> page text.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Embeddings root; output goes to <output-root>/<model-name>/.')
    parser.add_argument('--model', default=spec.default_model, help='Hugging Face model id or local path.')
    parser.add_argument('--model-name', default=None,
                        help='Output subdirectory name (default: last component of --model).')

    parser.add_argument('--normalization', choices=sorted(NORMALIZATIONS), default='ocr_words',
                        help='ocr_words: the words of the page, one space between them, then --max-chars; '
                             'ocr_lines: the same with the non-blank lines kept; none: the dump text unchanged.')
    parser.add_argument('--max-chars', type=int, default=MAX_CHARS,
                        help='Characters of normalized text kept (the dump itself is already capped at 4000).')
    parser.add_argument('--max-length', type=int, default=spec.default_max_length,
                        help='Max tokens per page including special tokens; longer pages keep their head and '
                             'final special token (counted as truncated in progress.tsv). '
                             + spec.max_length_help)
    parser.add_argument('--max-tokens-per-batch', type=int, default=spec.default_max_tokens_per_batch,
                        help='Padded token budget of one forward pass (batch size x longest sequence).')
    parser.add_argument('--max-batch-size', type=int, default=256,
                        help='Upper bound on sequences per forward pass, for very short pages.')
    parser.add_argument('--chunk-size', type=int, default=8192,
                        help='Pages per LMDB commit / progress line. Pages are sorted by length within a '
                             'chunk, so larger chunks pad less.')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16',
                        help='Storage dtype of the embedding vectors.')
    parser.add_argument('--compute-dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16',
                        help='Autocast dtype of the forward pass.')
    parser.add_argument('--attn-implementation', default=None,
                        choices=['flash_attention_2', 'sdpa', 'eager'],
                        help='Passed to from_pretrained (default: the model script\'s choice, else flash_attention_2 '
                             'if flash-attn is installed, else sdpa).')

    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after this many pages in this run (for testing); resumable as usual.')
    parser.add_argument('--map-size-gb', type=int, default=300,
                        help='Output LMDB map_size in GB (address-space reservation, not preallocation). '
                             '16.7M float16 1024-d vectors are ~34 GB of values.')
    parser.add_argument('--logging-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    add_gpu_arguments(parser, multi_gpu=True)
    spec.add_arguments(parser)

    return parser.parse_args(argv)


def iter_source(source_env: lmdb.Environment, after_key: Optional[bytes],
                limit: Optional[int]) -> Iterator[Tuple[bytes, str]]:
    """Yields (key, text) in key order, strictly after after_key."""
    produced = 0
    with source_env.begin(buffers=False) as txn:
        cursor = txn.cursor()
        positioned = cursor.set_range(after_key) if after_key is not None else cursor.first()
        if not positioned:
            return
        if after_key is not None and cursor.key() == after_key:
            if not cursor.next():
                return
        for key, value in cursor:
            if limit is not None and produced >= limit:
                return
            yield key, value.decode('utf-8')
            produced += 1


def iter_chunks(source_env: lmdb.Environment, tokenizer, after_key: Optional[bytes], limit: Optional[int],
                chunk_size: int, max_length: int,
                text_transform: Optional[Callable[[str], str]] = None,
                normalize: Callable[[str], str] = lambda text: text) -> Iterator[Chunk]:
    keys: List[bytes] = []
    texts: List[str] = []
    empty = 0

    def tokenize() -> Chunk:
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False,
                            return_attention_mask=False)['input_ids']
        truncated = 0
        input_ids = []
        for ids in encoded:
            if len(ids) > max_length:
                # Keep the head of the page and its final special token ([SEP] / end of text).
                ids = ids[:max_length - 1] + ids[-1:]
                truncated += 1
            input_ids.append(ids)
        return Chunk(list(keys), input_ids, empty, truncated)

    for key, text in iter_source(source_env, after_key, limit):
        keys.append(key)
        # Pages without text all get the same embedding, whatever whitespace their OCR left behind.
        text = normalize(text)
        if not text.strip():
            text = ''
            empty += 1
        texts.append(text_transform(text) if text_transform is not None else text)
        if len(keys) >= chunk_size:
            yield tokenize()
            keys, texts, empty = [], [], 0
    if keys:
        yield tokenize()


def prefetch(iterator: Iterator, depth: int = 2) -> Iterator:
    """Runs a CPU-bound iterator (LMDB read + tokenization) in a thread, ahead of the GPU loop."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # re-raised in the consumer
            q.put(exc)
        q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def make_batches(lengths: List[int], max_tokens_per_batch: int, max_batch_size: int) -> List[List[int]]:
    """Groups indices, longest first, so each batch's padded size stays under the token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: List[List[int]] = []
    current: List[int] = []
    for i in order:
        # order is descending, so the batch's first element sets its padded length
        longest = lengths[current[0]] if current else lengths[i]
        if current and (len(current) + 1 > max_batch_size or (len(current) + 1) * longest > max_tokens_per_batch):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def embed_chunk(model, pool, chunk: Chunk, pad_token_id: int, device: torch.device, compute_dtype: torch.dtype,
                max_tokens_per_batch: int, max_batch_size: int) -> np.ndarray:
    lengths = [len(ids) for ids in chunk.input_ids]
    out = np.empty((len(lengths), model.config.hidden_size), dtype=np.float32)
    for batch in make_batches(lengths, max_tokens_per_batch, max_batch_size):
        seq_len = lengths[batch[0]]
        input_ids = torch.full((len(batch), seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), seq_len), dtype=torch.long)
        for row, i in enumerate(batch):
            input_ids[row, :lengths[i]] = torch.tensor(chunk.input_ids[i], dtype=torch.long)
            attention_mask[row, :lengths[i]] = 1
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=compute_dtype,
                            enabled=compute_dtype != torch.float32):
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out[batch] = pool(hidden, attention_mask).cpu().numpy()
    return out


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean over tokens, in float32."""
    # Padded positions can come out NaN (fully masked attention rows), and NaN * 0 is NaN,
    # so they are overwritten rather than multiplied away.
    mask = attention_mask.unsqueeze(-1).bool()
    summed = hidden.to(torch.float32).masked_fill(~mask, 0.0).sum(dim=1)
    return summed / mask.sum(dim=1).clamp(min=1).to(torch.float32)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of each sequence's last real token (right padding), in float32."""
    last = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last].to(torch.float32)


class DevicePool:
    """One model replica per device, each served by a daemon thread taking chunks from a shared queue.

    Daemon threads, so an aborting run does not wait for the chunks still on the GPUs.
    """

    def __init__(self, replicas: List[Tuple[torch.nn.Module, torch.device]],
                 embed: Callable[[torch.nn.Module, torch.device, Chunk], object]):
        self.tasks: queue.Queue = queue.Queue()
        for model, device in replicas:
            threading.Thread(target=self._serve, args=(model, device, embed), daemon=True).start()

    def _serve(self, model, device, embed):
        if device.type == 'cuda':
            torch.cuda.set_device(device)  # the current device is per thread; kernels launch on it
        while True:
            future, chunk = self.tasks.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(embed(model, device, chunk))
            except BaseException as exc:  # re-raised by future.result() in the committing thread
                future.set_exception(exc)

    def submit(self, chunk: Chunk) -> Future:
        future: Future = Future()
        self.tasks.put((future, chunk))
        return future


def select_devices(n_gpu: int) -> List[torch.device]:
    if not torch.cuda.is_available():
        if n_gpu > 1:
            raise SystemExit(f'--n-gpu {n_gpu} requested, but CUDA is not available.')
        return [torch.device('cpu')]
    if n_gpu > torch.cuda.device_count():
        raise SystemExit(f'--n-gpu {n_gpu} requested, but only {torch.cuda.device_count()} GPU(s) are visible '
                         f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}).')
    return [torch.device('cuda', i) for i in range(n_gpu)]


def run(spec: EncoderSpec, argv=None) -> None:
    args = parse_arguments(spec, argv)

    model_name = args.model_name or args.model.rstrip('/').split('/')[-1]
    output_dir = args.output_root / model_name
    lmdb_dir = output_dir / 'lmdb'
    progress_path = output_dir / 'progress.tsv'
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    lock_file = acquire_run_lock(output_dir)  # noqa: F841 -- held until the process exits

    setup_logging(output_dir / 'log.txt', args.logging_level)
    logger.info(' '.join(sys.argv))
    logger.info(f'torch {torch.__version__}, transformers {transformers.__version__}, '
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')

    devices = select_devices(args.n_gpu)
    logger.info(f'Devices: {", ".join(str(device) for device in devices)}')
    compute_dtype = getattr(torch, args.compute_dtype)

    attn_implementation = args.attn_implementation or spec.default_attn_implementation or (
        'flash_attention_2' if importlib.util.find_spec('flash_attn') is not None else 'sdpa')
    logger.info(f'Attention implementation: {attn_implementation}')

    tokenizer, model = spec.load(args.model, attn_implementation, compute_dtype)
    logger.info(f'Model weights dtype: {model.dtype}')
    model.eval()
    # Replicas are copied on the CPU, before the original moves to its own device.
    replicas = [(copy.deepcopy(model), device) for device in devices[1:]]
    replicas.insert(0, (model, devices[0]))
    for replica, device in replicas:
        replica.to(device)
    if args.max_length > model.config.max_position_embeddings:
        raise SystemExit(f'--max-length {args.max_length} exceeds the model limit '
                         f'{model.config.max_position_embeddings}.')

    source_env = lmdb.open(str(args.source_lmdb), readonly=True, lock=False, readahead=True)
    source_entries = source_env.stat()['entries']

    meta = {
        'model': args.model,
        'model_commit_hash': getattr(model.config, '_commit_hash', None),
        'pooling': spec.pooling,
        'normalized': False,
        'dtype': args.dtype,
        'dim': model.config.hidden_size,
        'max_length': args.max_length,
        'truncation': 'head',
        'weights_dtype': str(model.dtype).removeprefix('torch.'),
        'compute_dtype': args.compute_dtype,
        'empty_text': 'whitespace-only text is encoded as ""',
        'normalization': args.normalization,
        'max_chars': args.max_chars if args.normalization != 'none' else None,
        **spec.extra_meta(args),
        'source_lmdb': str(args.source_lmdb.resolve()),
        'source_entries': source_entries,
        'key_format': '{library}_{page_id}, same as the source LMDB',
        'value_format': f'raw {args.dtype} bytes, np.frombuffer(value, dtype="{args.dtype}")',
    }
    check_or_write_meta(output_dir / 'meta.json', meta, FIXED_META_FIELDS + spec.extra_fixed_fields)

    last_key, total_done, chunk_index = load_progress(progress_path, PROGRESS_HEADER)
    if last_key is not None:
        logger.info(f'Resuming after key {last_key.decode()} ({total_done}/{source_entries} done, '
                    f'chunk {chunk_index}).')
    else:
        logger.info(f'Starting from scratch, {source_entries} source entries.')

    output_env = lmdb.open(str(lmdb_dir), map_size=args.map_size_gb * 1024 ** 3)
    stop = StopRequest()
    store_dtype = np.dtype(args.dtype)
    run_done = 0
    run_start = time.time()

    def embed(replica, device, chunk: Chunk) -> Tuple[np.ndarray, float, torch.device]:
        chunk_start = time.time()
        embeddings = embed_chunk(replica, spec.pool, chunk, tokenizer.pad_token_id, device, compute_dtype,
                                 args.max_tokens_per_batch, args.max_batch_size).astype(store_dtype)
        return embeddings, time.time() - chunk_start, device

    pool = DevicePool(replicas, embed)

    def commit(chunk: Chunk, future: Future) -> None:
        nonlocal total_done, run_done, chunk_index
        embeddings, chunk_seconds, device = future.result()
        bad_rows = np.flatnonzero(~np.isfinite(embeddings).all(axis=1))
        if bad_rows.size:
            raise RuntimeError(f'{bad_rows.size} non-finite embeddings in chunk {chunk_index}, e.g. key '
                               f'{chunk.keys[bad_rows[0]].decode()}; nothing of this chunk was stored.')

        with output_env.begin(write=True) as txn:
            for key, vector in zip(chunk.keys, embeddings):
                txn.put(key, vector.tobytes())

        total_done += len(chunk.keys)
        run_done += len(chunk.keys)
        # the rate of the one device that embedded this chunk; the run average covers all of them
        docs_per_s = len(chunk.keys) / max(chunk_seconds, 1e-9)
        append_progress(progress_path, (
            time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), chunk_index,
            chunk.keys[0].decode(), chunk.keys[-1].decode(), len(chunk.keys),
            chunk.empty, chunk.truncated, total_done, f'{docs_per_s:.1f}'), PROGRESS_HEADER)

        run_rate = run_done / max(time.time() - run_start, 1e-9)
        eta_h = (source_entries - total_done) / max(run_rate, 1e-9) / 3600
        logger.info(f'chunk {chunk_index}: {len(chunk.keys)} pages ({chunk.truncated} truncated, '
                    f'{chunk.empty} empty), {docs_per_s:.1f} pages/s on {device} | total '
                    f'{total_done}/{source_entries} ({100 * total_done / source_entries:.2f}%), '
                    f'run avg {run_rate:.1f} pages/s, ETA {eta_h:.1f} h')
        chunk_index += 1

    try:
        chunks = prefetch(iter_chunks(source_env, tokenizer, last_key, args.limit, args.chunk_size,
                                      args.max_length, spec.make_text_transform(args),
                                      functools.partial(NORMALIZATIONS[args.normalization],
                                                        max_chars=args.max_chars)))
        pending: collections.deque = collections.deque()  # (chunk, future) in key order
        for chunk in chunks:
            pending.append((chunk, pool.submit(chunk)))
            # Two chunks per device in flight, so no device idles while an older chunk is committed.
            if len(pending) >= 2 * len(devices):
                commit(*pending.popleft())
            if stop.requested:
                break
        while pending:
            chunk, future = pending.popleft()
            # On a stop, chunks no device has started are dropped. Two threads may start chunks out of order,
            # so everything after the first dropped chunk is dropped too: commits stay a prefix in key order.
            if stop.requested and future.cancel():
                for _, later in pending:
                    later.cancel()
                break
            commit(chunk, future)
        if stop.requested:
            logger.info('Stopped on request; rerun the same command to resume.')
        output_env.sync()
    finally:
        output_env.close()
        source_env.close()

    logger.info(f'Run finished: {run_done} pages this run, {total_done}/{source_entries} total.')
