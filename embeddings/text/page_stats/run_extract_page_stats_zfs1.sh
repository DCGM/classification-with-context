#!/bin/bash
# Page statistics of all zfs1 pages (the "private" storage) -> <data>/2026-09-16.db_page_stats_dump.private.
# Run by the user, with zfs1 mounted at the path in the DB (/mnt/zfs1/library-downloads/...).
# zfs1 has only <doc>.page_xml.zip: each page's PAGE XML is converted to ALTO in memory (PERO, words placed
# by length) and only the sparse statistics vector is stored. No text or XML is written or logged.
# Resumable: rerun to continue. Extra arguments go to extract_to_lmdb.py, e.g.
#   --max-zips 20          a trial on the first 20 documents (then rerun without it; done pages are skipped)
#   --workers 8            worker processes (default here 10; keep it at most 10)
#   --path-map /mnt/zfs1=/media/zfs1   if zfs1 is mounted elsewhere
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
OUT_DIR="$DATA/2026-09-16.db_page_stats_dump.private"
PYTHON=/home/ikohut/python_env/pero-ocr/bin/python3
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
"$PYTHON" -m embeddings.text.page_stats.extract_to_lmdb \
    --mapping-file "$DATA/2026-09-16.db_text_dump/mapping.tsv" \
    --path-prefix /mnt/zfs1 \
    --lmdb-dir "$OUT_DIR/lmdb" \
    --progress-file "$OUT_DIR/progress.tsv" \
    --log-file "$OUT_DIR/log.txt" \
    --workers 10 \
    "$@"
