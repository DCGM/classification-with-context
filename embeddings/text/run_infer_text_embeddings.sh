#!/bin/bash
# Page-text embeddings of the text dump, the text reduced to its words (no layout).
# Usage: run_infer_text_embeddings.sh <mmbert|qwen3> <public|private> [extra infer arguments, e.g. --n-gpu 2 --limit 1000]
# Output: <kolosus>/digiknihovna.data/<part>/text/<Model>.ocr_words/, next to the earlier embeddings.
# Resumable: rerun the same command to continue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/ikohut/python_env/metakat/bin/python}"
SSD=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
KOLOSUS=/mnt/kolosus/data
ENGINE_MODELS="$KOLOSUS/metakat_worker/src_engines/chapter.v1.7.0/page_type/core/page_type_core_engine_context.2026-10-28"

model="$1"
part="$2"
shift 2
case "$part" in
    public) source="$SSD/2026-09-16.db_text_dump/lmdb" ;;
    private) source="$KOLOSUS/digiknihovna.data/private/db_text_dump.2026-09-16/lmdb" ;;
    *) echo "part must be public or private" >&2; exit 2 ;;
esac
case "$model" in
    # the mmBERT weights the MetaKat engine runs
    mmbert) module=embeddings.text.mmbert.infer_mmbert; arguments=(--model "$ENGINE_MODELS/mmbert" --model-name mmBERT-base.ocr_words) ;;
    qwen3) module=embeddings.text.qwen3.infer_qwen3; arguments=(--model-name Qwen3-Embedding-0.6B.ocr_words) ;;
    *) echo "model must be mmbert or qwen3" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
exec "$PYTHON" -m "$module" \
    --source-lmdb "$source" \
    --output-root "$KOLOSUS/digiknihovna.data/$part/text" \
    "${arguments[@]}" "$@"
