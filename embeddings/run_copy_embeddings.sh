#!/bin/bash
# Compact copies of the mmBERT and DINOv2 embedding LMDBs (public and private) from kolosus to the SSD,
# next to the text and page statistics dumps: <data>/2026-09-16.db_embedding_dump/{public,private}/...
# One after another (they read the same disks). Resumable: rerun to continue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC=/mnt/kolosus/data/digiknihovna.data
OUT=/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_embedding_dump
PYTHON=/home/ikohut/python_env/metakat/bin/python3

cd "$REPO_ROOT"
for part in public private; do
    for model in text/mmBERT-base image/vitb14_dinov2_2026-09-20_public_private_2; do
        "$PYTHON" -m embeddings.copy_lmdb_compact "$SRC/$part/$model" "$OUT/$part/$model"
    done
done
