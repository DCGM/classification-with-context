# Rules for working in this repo

## Private page text

Never read, print, sample, grep or otherwise explore the source text of the **private** text dump
(`/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/2026-09-16.db_text_dump.private/`,
its `lmdb/` and any copy of it). Embeddings of private pages are fine to read.

Text analysis (layout statistics, TF-IDF, ...) is developed and debugged on the public text dump
(`.../data/2026-09-16.db_text_dump/lmdb`) only. On the private text a finished script may only be
*run*, and only when its outputs are aggregates or predictions: metrics, counts, predicted labels.
It must never write or log text, snippets, tokens, vocabularies or per-page text features, and its
logs are not to be inspected for text. `embedding_probe/text_baselines.py` reads the private text
only when `--private-text-lmdb` is given explicitly.

## zfs1 (private library storage)

Never access `/mnt/zfs1` in any way: no listing, no existence checks (`test -e`, `stat`, `find`),
no inspecting the mount, no reading files. Scripts for it (e.g. `run_fetch_private_alto.sh`) are run
by the user; diagnose only from the DB metadata and what the user reports.

## matylda0

`/mnt/matylda0` is under the user's supervision: read only, never write there. Working files go to
`/home/ikohut/tmp/`.
