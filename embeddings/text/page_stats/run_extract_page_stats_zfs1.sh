#!/bin/bash
# Page statistics of all zfs1 pages (the "private" storage). Run by the user, on a machine with zfs1 mounted
# at the path in the DB (/mnt/zfs1/library-downloads/...). The output, by default next to the data on zfs1:
#   $OUT_DIR = /mnt/zfs1/library-downloads/2026-09-16.db_page_stats_dump.private/{lmdb,progress.tsv,log.txt}
# move_zfs1_progress_to_zfs1.sh copies a run started elsewhere there (with mapping.tsv), and this script
# continues it. Afterwards copy $OUT_DIR back to <SSD data>/2026-09-16.db_page_stats_dump.private.
#
# zfs1 has only <doc>.page_xml.zip: each page's PAGE XML is converted to ALTO in memory (PERO, words placed
# by length) and only the sparse statistics vector is stored. No text or XML is written or logged.
#
# Settings (environment variables):
#   PYTHON    a python with pero-ocr, lxml, lmdb and numpy (default: the pero-ocr venv of this machine)
#   OUT_DIR   output directory (default above)
#   MAPPING   the text dump's mapping.tsv (default: $OUT_DIR/mapping.tsv)
# Extra arguments go to extract_to_lmdb.py, e.g.
#   --max-zips 20          a trial on the first 20 documents (then rerun without it; done pages are skipped)
#   --workers 8            worker processes (default here 10; keep it at most 10)
#   --path-map /mnt/zfs1=/media/zfs1   if zfs1 is mounted elsewhere
# Resumable: rerun to continue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/ikohut/python_env/pero-ocr/bin/python3}"
OUT_DIR="${OUT_DIR:-/mnt/zfs1/library-downloads/2026-09-16.db_page_stats_dump.private}"
MAPPING="${MAPPING:-$OUT_DIR/mapping.tsv}"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
"$PYTHON" -m embeddings.text.page_stats.extract_to_lmdb \
    --mapping-file "$MAPPING" \
    --path-prefix /mnt/zfs1 \
    --lmdb-dir "$OUT_DIR/lmdb" \
    --progress-file "$OUT_DIR/progress.tsv" \
    --log-file "$OUT_DIR/log.txt" \
    --workers 10 \
    "$@"
