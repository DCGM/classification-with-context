"""Compute mmBERT-base page text embeddings for a page-text LMDB dump.

Output layout, resume and progress are described in embeddings/text/inference.py.

This is the computation of MetaKat's MmbertEncoder (metakat/page_type/engines/core/context/mmbert.py) on the
text given to it: float32 weights under bfloat16 autocast, flash attention
(flash-attn when installed, else sdpa, as the engine chooses), the padding-NaN patch, the
attention-masked mean of the last hidden state, stored as float16. The text is reduced to its words
(--normalization ocr_words, the default; the first embeddings were computed with "none"), and the output
directory is named after --model-name, which must differ from the earlier embeddings' directory.

The embedding is the attention-masked mean of the last hidden state. mmBERT is
an MLM encoder, not a trained sentence embedder; the mean is the conventional
pooling for it.

Requires transformers with ModernBERT support (>=4.48). The model repository ships only pytorch_model.bin,
which transformers>=4.50 refuses to load on torch<2.6. With flash-attn installed ModernBERT unpads each batch and
runs varlen attention, which is the fast path (about 1.6x the padded sdpa one here); --attn-implementation chooses
another. The padded sdpa path needs patch_padding_nan.

Example:
    python -m embeddings.text.mmbert.infer_mmbert \\
        --source-lmdb /data/2026-09-16.db_text_dump/lmdb \\
        --output-root /data/2026-09-16.db_text_dump/embeddings \\
        --model-name mmBERT-base.ocr_words
"""

import sys

from embeddings.gpu_bootstrap import bootstrap_gpus


# CUDA_VISIBLE_DEVICES must be finalized before importing PyTorch or Transformers.
if __name__ == '__main__':
    bootstrap_gpus(sys.argv[1:], multi_gpu=True)


import torch
from transformers import AutoModel, AutoTokenizer

from embeddings.text.inference import EncoderSpec, mean_pool, run


def patch_padding_nan(model) -> None:
    """
    Keeps padded batches from turning real tokens into NaN (ModernBERT sdpa/eager, transformers 4.49-4.57).

    A padding query farther than local_attention/2 from every real token has all its keys masked by
    finfo.min; inside attention score + finfo.min overflows to -inf, the row's softmax is NaN, and the
    next layer spreads it into real tokens through NaN * 0 in attention @ values. Unmasking the diagonal
    lets every query attend at least to itself: real rows already had it unmasked, so their outputs are
    unchanged, and padding rows stay finite.
    """
    original = model._update_attention_mask

    def update_attention_mask(attention_mask, output_attentions):
        global_mask, sliding_window_mask = original(attention_mask, output_attentions=output_attentions)
        diagonal = torch.eye(global_mask.shape[-1], dtype=torch.bool, device=global_mask.device)
        return global_mask.masked_fill(diagonal, 0.0), sliding_window_mask.masked_fill(diagonal, 0.0)

    model._update_attention_mask = update_attention_mask


def load(model_id: str, attn_implementation: str, compute_dtype: torch.dtype):
    # float32 weights under autocast (compute_dtype unused), as the mmBERT-base embeddings were computed
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, attn_implementation=attn_implementation)
    patch_padding_nan(model)
    return tokenizer, model


SPEC = EncoderSpec(
    description='Compute mmBERT page text embeddings for every entry of a page-text LMDB.',
    default_model='jhu-clsp/mmBERT-base',
    pooling='mean_last_hidden_state',
    load=load,
    pool=mean_pool,
    default_max_length=8192,
    max_length_help='The default is the model limit, which page texts capped at 4000 chars never reach '
                    '(~1300 tokens median, ~4600 max).',
)


def main(argv=None):
    run(SPEC, argv)


if __name__ == '__main__':
    main()
