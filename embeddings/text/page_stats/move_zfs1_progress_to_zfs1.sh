#!/bin/bash
# Moves the zfs1 page statistics extraction to another machine: copies its current progress from the SSD
# (LMDB, progress.tsv, log.txt) and the mapping file to zfs1, where run_extract_page_stats_zfs1.sh then
# continues (it skips the pages already in progress.tsv). Run by the user, on the machine with the SSD,
# after stopping the local zfs1 extraction.
#
#   embeddings/text/page_stats/move_zfs1_progress_to_zfs1.sh [target dir]
#
# Default target: /mnt/zfs1/library-downloads/2026-09-16.db_page_stats_dump.private
set -euo pipefail

SSD=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
SRC="$SSD/2026-09-16.db_page_stats_dump.private"
MAPPING="$SSD/2026-09-16.db_text_dump/mapping.tsv"
DST="${1:-/mnt/zfs1/library-downloads/2026-09-16.db_page_stats_dump.private}"

if pgrep -f "extract_to_lmdb.*--path-prefix /mnt/zfs1" > /dev/null; then
    echo "The zfs1 extraction is still running here: stop it first (Ctrl+C, or kill its process group)." >&2
    exit 1
fi
if [ -e "$DST/lmdb/data.mdb" ]; then
    echo "$DST/lmdb/data.mdb exists already: not overwriting it." >&2
    exit 1
fi

mkdir -p "$DST/lmdb"
cp -v "$SRC/lmdb/data.mdb" "$DST/lmdb/data.mdb"
cp -v "$SRC/progress.tsv" "$SRC/log.txt" "$DST/"
cp -v "$MAPPING" "$DST/mapping.tsv"

for f in lmdb/data.mdb progress.tsv; do
    if [ "$(stat -c %s "$SRC/$f")" != "$(stat -c %s "$DST/$f")" ]; then
        echo "size mismatch for $f" >&2
        exit 1
    fi
done
echo
echo "Copied $(wc -l < "$DST/progress.tsv") progress lines to $DST."
echo "On the other machine (repository pulled, branch embedding-experiments):"
echo "  PYTHON=<venv>/bin/python embeddings/text/page_stats/run_extract_page_stats_zfs1.sh"
echo "When it has finished, copy $DST back to $SRC (e.g. rsync -a)."
