#!/bin/bash
# Fetches the ALTO XML of the private pages the experiments need (the annotated pages with their
# context windows up to K = 3, see write_alto_page_list.py) into an LMDB next to the private text
# LMDB: <private text dump>/alto (key {library}_{page uuid}, value the zlib-compressed ALTO XML).
#
# In the DB (meta_records.image_path) these pages lie under /mnt/zfs1/library-downloads/{mzk,knav}
# (92,967 pages, ~11.5k document zips <docdir>.alto_xml.zip). Mount that storage, set PRIVATE_MOUNT
# to where it is mounted here, and run:
#
#   embedding_probe/run_fetch_private_alto.sh --dry-run   # DB lookup only: page counts per path prefix
#   embedding_probe/run_fetch_private_alto.sh             # the fetch; a rerun resumes
#
# Needs the DB tunnel on localhost:5888 and the db_text_dump venv (lmdb, psycopg2). Progress and
# counts go to $OUT/log.txt and $OUT/progress.tsv.
set -euo pipefail

PRIVATE_MOUNT="${PRIVATE_MOUNT:-/mnt/zfs1/library-downloads}"   # where the storage is mounted here
DB_PREFIX=/mnt/zfs1/library-downloads                           # how the DB paths start
DATA=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data
PAGES="$DATA/embedding_probe/alto_pages.tsv"
OUT="$DATA/2026-09-16.db_text_dump.private/alto"
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
"$PYTHON" -m embedding_probe.fetch_alto \
    --pages "$PAGES" \
    "${MAP[@]}" \
    --path-prefix "$PRIVATE_MOUNT" \
    --out-lmdb "$OUT" \
    "$@"
