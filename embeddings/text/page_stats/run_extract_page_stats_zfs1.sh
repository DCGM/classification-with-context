#!/bin/bash
# Page statistics of all zfs1 pages (the "private" storage). Run by the user, on a machine with zfs1 mounted
# at the path in the DB (/mnt/zfs1/library-downloads/...). Starts from scratch if $OUT_DIR has no progress
# yet, otherwise continues (done pages are skipped). Output, by default next to the data on zfs1:
#   $OUT_DIR = /mnt/zfs1/library-downloads/2026-09-16.db_page_stats_dump.private/{lmdb,progress.tsv,log.txt}
# Afterwards copy $OUT_DIR back to <SSD data>/2026-09-16.db_page_stats_dump.private.
#
# zfs1 has only <doc>.page_xml.zip: each page's PAGE XML is read with embeddings/ocr_xml.py (words placed by
# length) and only the sparse statistics vector is stored. No text or XML is written or logged.
#
# Settings (environment variables):
#   PYTHON    a python with this repository installed (pip install -e .), default python3
#   OUT_DIR   output directory (default above)
#   MAPPING   the text dump's mapping.tsv; default: the first that exists of $OUT_DIR/mapping.tsv and
#             <SSD data>/2026-09-16.db_text_dump/mapping.tsv
# Extra arguments go to extract_to_lmdb.py, e.g.
#   --max-zips 20          a trial on the first 20 documents (then rerun without it; done pages are skipped)
#   --workers 8            worker processes (default here 10; keep it at most 10)
#   --path-map /mnt/zfs1=/media/zfs1   if zfs1 is mounted elsewhere
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-/mnt/zfs1/library-downloads/2026-09-16.db_page_stats_dump.private}"
SSD_MAPPING=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump/mapping.tsv
if [ -z "${MAPPING:-}" ]; then
    for candidate in "$OUT_DIR/mapping.tsv" "$SSD_MAPPING"; do
        if [ -f "$candidate" ]; then MAPPING="$candidate"; break; fi
    done
fi

cd "$REPO_ROOT"
if ! "$PYTHON" -c "import numpy, lmdb, embeddings.text.page_stats" 2> /dev/null; then
    echo "$PYTHON cannot import numpy, lmdb and this repository. Create the environment and set PYTHON:" >&2
    echo "  python3 -m venv ~/python_env/classification-with-context" >&2
    echo "  ~/python_env/classification-with-context/bin/pip install -e \"$REPO_ROOT[dev]\" numpy==2.5.4 lmdb==3.0.0" >&2
    echo "  PYTHON=~/python_env/classification-with-context/bin/python $0 $*" >&2
    exit 1
fi
if [ -n "${MAPPING:-}" ] && [ ! -f "$MAPPING" ]; then
    echo "MAPPING=$MAPPING does not exist." >&2
    exit 1
fi
if [ -z "${MAPPING:-}" ]; then
    echo "No mapping file: neither $OUT_DIR/mapping.tsv nor $SSD_MAPPING exists." >&2
    echo "Copy the text dump's mapping.tsv to $OUT_DIR/ or run with MAPPING=/path/to/mapping.tsv" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
done_pages=0
if [ -f "$OUT_DIR/progress.tsv" ]; then done_pages=$(wc -l < "$OUT_DIR/progress.tsv"); fi
echo "PYTHON=$PYTHON"
echo "MAPPING=$MAPPING"
echo "OUT_DIR=$OUT_DIR ($done_pages pages recorded before)"

exec "$PYTHON" -m embeddings.text.page_stats.extract_to_lmdb \
    --mapping-file "$MAPPING" \
    --path-prefix /mnt/zfs1 \
    --lmdb-dir "$OUT_DIR/lmdb" \
    --progress-file "$OUT_DIR/progress.tsv" \
    --log-file "$OUT_DIR/log.txt" \
    --workers 10 \
    "$@"
