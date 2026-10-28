"""Compute Qwen3-Embedding-0.6B page text embeddings for a page-text LMDB dump.

Output layout, resume and progress are described in embeddings/text/inference.py. The text is reduced to its words
(--normalization ocr_words, the default; "none" reproduces the first Qwen3 embeddings); use a --model-name that differs from the earlier embeddings' directory.

Qwen3-Embedding is a decoder trained as a text embedder: the embedding is the
hidden state of the last token, which the tokenizer appends itself
(<|endoftext|>). Batches are right-padded, which is safe for a causal model:
real tokens never attend to the padding after them. Weights are loaded in the
compute dtype (bf16 by default). Vectors are stored unnormalized; the model
card L2-normalizes before cosine similarity.

Throughput on an RTX 3090 with flash-attention is ~28 pages/s (pages capped at
4000 chars are ~1400 tokens median), i.e. about a week for 16.7M pages.
--n-gpu N runs one replica per GPU (the first N of CUDA_VISIBLE_DEVICES, or
N GPUs claimed with --use-safe-gpu) into the same output.

Pages are embedded as documents, i.e. without a prompt, as the model card does
for retrieval documents. --instruction adds the query-style prefix
"Instruct: <instruction>\\nQuery:" instead, for task-specific embeddings; it is
recorded in meta.json and must match on resume.

Example:
    python -m embeddings.text.qwen3.infer_qwen3 \\
        --source-lmdb /data/2026-09-16.db_text_dump/lmdb \\
        --output-root /data/2026-09-16.db_text_dump/embeddings \\
        --model-name Qwen3-Embedding-0.6B.ocr_words --n-gpu 4
"""

import sys

from embeddings.gpu_bootstrap import bootstrap_gpus


# CUDA_VISIBLE_DEVICES must be finalized before importing PyTorch or Transformers.
if __name__ == '__main__':
    bootstrap_gpus(sys.argv[1:], multi_gpu=True)


import torch
from transformers import AutoModel, AutoTokenizer

from embeddings.text.inference import EncoderSpec, last_token_pool, run


def load(model_id: str, attn_implementation: str, compute_dtype: torch.dtype):
    # Weights in the compute dtype (bf16 by default), as the model card runs it: with float32 weights the
    # residual stream stays float32 under autocast, doubling activation memory for no measurable gain.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, attn_implementation=attn_implementation, dtype=compute_dtype)
    return tokenizer, model


def add_arguments(parser):
    parser.add_argument('--instruction', default=None,
                        help='Embed each page as "Instruct: <instruction>\\nQuery:<page text>" instead of as a '
                             'plain document (default: no instruction).')


def make_text_transform(args):
    if args.instruction is None:
        return None
    prefix = f'Instruct: {args.instruction}\nQuery:'
    return lambda text: prefix + text


SPEC = EncoderSpec(
    description='Compute Qwen3-Embedding page text embeddings for every entry of a page-text LMDB.',
    default_model='Qwen/Qwen3-Embedding-0.6B',
    pooling='last_token',
    load=load,
    pool=last_token_pool,
    default_max_length=8192,
    # ~3.3 GiB peak with bf16 weights and flash-attention; larger budgets are not faster (compute-bound)
    default_max_tokens_per_batch=16384,
    max_length_help='The default is the model card\'s 8192 (the model accepts 32768); page texts capped at '
                    '4000 chars stay well below it.',
    add_arguments=add_arguments,
    make_text_transform=make_text_transform,
    extra_meta=lambda args: {'instruction': args.instruction},
    extra_fixed_fields=('instruction',),
)


def main(argv=None):
    run(SPEC, argv)


if __name__ == '__main__':
    main()
