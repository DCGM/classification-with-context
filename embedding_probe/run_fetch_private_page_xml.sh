#!/bin/bash
# Fetches the PAGE XML of the experiment pages stored on zfs1 (the annotated pages with their context
# windows up to K = 3, see write_ocr_page_list.py) from <docdir>.page_xml.zip into an LMDB next to
# the private text LMDB: <private text dump>/page_xml (key {library}_{page uuid}, value the
# zlib-compressed PAGE XML). zfs1 has no ALTO zips; the PAGE XML is converted to ALTO afterwards by
# convert_page_xml_to_alto.py into <private text dump>/alto.
#
# The split is by storage, not by the DB's public/private flag (the storage paths predate an
# archival update): everything on matylda0 has ALTO in <public text dump>/alto, everything on zfs1
# goes here. In the DB (meta_records.image_path) the zfs1 pages lie under
# /mnt/zfs1/library-downloads/{mzk,knav} (92,967 pages, ~11.5k documents). Mount that storage
# (PRIVATE_MOUNT defaults to the DB path) and run:
#
#   embedding_probe/run_fetch_private_page_xml.sh --dry-run   # DB lookup only: page counts per path prefix
#   embedding_probe/run_fetch_private_page_xml.sh             # the fetch; a rerun resumes
#
# Needs the DB tunnel on localhost:5888 and the db_text_dump venv (lmdb, psycopg2). Progress and
# counts go to $OUT/log.txt and $OUT/progress.tsv (counts and paths only); --diagnose N logs the
# names of the files next to the first N missing zips.
set -euo pipefail

PRIVATE_MOUNT="${PRIVATE_MOUNT:-/mnt/zfs1/library-downloads}"   # where the storage is mounted here
DB_PREFIX=/mnt/zfs1/library-downloads                           # how the DB paths start
DATA=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
PAGES="$DATA/embedding_probe/ocr_pages.tsv"
OUT="$DATA/2026-09-16.db_text_dump.private/page_xml"
PYTHON=/home/ikohut/python_env/db_text_dump/bin/python3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" != "--dry-run" && ! -d "$PRIVATE_MOUNT/mzk" ]]; then
    echo "$PRIVATE_MOUNT/mzk not found: mount the private storage and set PRIVATE_MOUNT" >&2
    exit 1
fi
MAP=()
if [[ "$PRIVATE_MOUNT" != "$DB_PREFIX" ]]; then
    MAP=(--path-map "$DB_PREFIX=$PRIVATE_MOUNT")
fi
cd "$REPO"
"$PYTHON" -m embedding_probe.fetch_ocr_xml \
    --kind page_xml \
    --pages "$PAGES" \
    "${MAP[@]}" \
    --path-prefix "$PRIVATE_MOUNT" \
    --out-lmdb "$OUT" \
    "$@"
