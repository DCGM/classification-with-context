#!/bin/bash
# Page-text embeddings of the text dump, the text without layout noise (see embeddings/text/normalization.py).
# Usage: run_infer_text_embeddings.sh <mmbert|qwen3> <public|private> [extra infer arguments, e.g. --n-gpu 2 --limit 1000]
# Environment: NORMALIZATION (ocr_lines, ocr_words or none; default ocr_lines: the OCR lines without the blank ones),
#   SOURCE (the text LMDB), OUT_ROOT (the output goes to $OUT_ROOT/<Model>-base.<NORMALIZATION>/, a directory with
#   the layout of the earlier embeddings: lmdb/, meta.json, progress.tsv, log.txt).
# Defaults: public reads the SSD copy of the text dump and writes next to it; private reads the kolosus copy
# (sequentially, the page text is a trickle next to the GPU's speed) and writes next to the SSD private dump.
# Resumable: rerun the same command to continue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/ikohut/python_env/metakat/bin/python}"
SSD=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
KOLOSUS=/mnt/kolosus/data
ENGINE_MODELS="$KOLOSUS/metakat_worker/src_engines/chapter.v1.7.0/page_type/core/page_type_core_engine_context.2026-10-28"
NORMALIZATION="${NORMALIZATION:-ocr_lines}"

model="$1"
part="$2"
shift 2
case "$part" in
    public) source="${SOURCE:-$SSD/2026-09-16.db_text_dump/lmdb}"; out_root="${OUT_ROOT:-$SSD/2026-09-16.db_text_dump/embeddings}" ;;
    private) source="${SOURCE:-$KOLOSUS/digiknihovna.data/private/db_text_dump.2026-09-16/lmdb}"; out_root="${OUT_ROOT:-$SSD/2026-09-16.db_text_dump.private/embeddings}" ;;
    *) echo "part must be public or private" >&2; exit 2 ;;
esac
case "$model" in
    # the mmBERT weights the MetaKat engine runs
    mmbert) module=embeddings.text.mmbert.infer_mmbert; arguments=(--model "$ENGINE_MODELS/mmbert" --model-name "mmBERT-base.$NORMALIZATION") ;;
    qwen3) module=embeddings.text.qwen3.infer_qwen3; arguments=(--model-name "Qwen3-Embedding-0.6B.$NORMALIZATION") ;;
    *) echo "model must be mmbert or qwen3" >&2; exit 2 ;;
esac

mkdir -p "$out_root"
cd "$REPO_ROOT"
exec "$PYTHON" -u -m "$module" \
    --source-lmdb "$source" \
    --output-root "$out_root" \
    --normalization "$NORMALIZATION" \
    "${arguments[@]}" "$@"
