#!/bin/bash
# Page statistics of all matylda0 pages (the "public" storage) -> <data>/2026-09-16.db_page_stats_dump.
# Streams the document zips (ALTO, else PAGE XML converted in memory); stores only the sparse vectors.
# Resumable: rerun to continue. Extra arguments go to extract_to_lmdb.py (e.g. --workers 8, --max-zips 100; keep at most 10 workers).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
OUT_DIR="$DATA/2026-09-16.db_page_stats_dump"
PYTHON=/home/ikohut/python_env/pero-ocr/bin/python3
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
"$PYTHON" -m embeddings.text.page_stats.extract_to_lmdb \
    --mapping-file "$DATA/2026-09-16.db_text_dump/mapping.tsv" \
    --path-prefix /mnt/matylda0 \
    --lmdb-dir "$OUT_DIR/lmdb" \
    --progress-file "$OUT_DIR/progress.tsv" \
    --log-file "$OUT_DIR/log.txt" \
    --workers 10 \
    "$@"
