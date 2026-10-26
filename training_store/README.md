# Document training store

This package converts the existing audited page index and vector LMDBs into an
immutable directory of document-contiguous packed records. It never reads source
text or OCR. The `page_sequence` runner can train and evaluate directly from this store.

Each 3,386-byte little-endian record contains the original 768-dimensional DINOv2
and mmBERT FP16 vectors, all 147 grouped/normalized/block-scaled statistics rounded
to FP16, logical row ID, document ID, original page order, three labels and modality
availability bits. Missing labels use -100. Labels, order and identifiers are metadata,
not input features. Document boundaries are determined from adjacent sampled document
IDs; the first element of a window has no boundary target.

`catalogue.sqlite` is an independent copy of the original audited index, including
hierarchy, splits, sampler anchors and exclusions. `documents.npy` maps document IDs
to original first row, length, shard and record offset. Original logical row IDs and
sampler behavior are preserved while physical document order follows library,
periodical, volume and document identity. Entire documents stay inside a shard.

Build with bounded feature buffers and a small worker count:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m training_store.build \
  --config /path/to/experiment.json \
  --sources /path/to/vector-sources.json \
  --output /path/to/nvme/document-store-fp16-v1 \
  --workers 1 --block-pages 256
```

`--sources` is optional and has the same format as the experiment's `features`
configuration. It can point at archived vector LMDBs, while validation checks against
the experiment's original sources. Statistics preprocessing and public/private
lookup precedence reuse the existing pipeline.

The converter copies and hashes the catalogue, freezes source signatures and feature
schema, validates every emitted vector for finite FP16 values, records availability
counts, and hashes each completed shard. A stratified correctness pilot can be built
with `--pilot-documents 256 --shard-mib 64`; this is not a cold-cache throughput test.

Rerun the same command to resume after an interruption. Completed shards are checked
and reused; an unfinished shard is rewritten. Changed code or inputs block resume.
An exclusive lock prevents concurrent writers. Failures leave no published manifest.
Only after all shards and reference comparisons pass is `manifest.json` published
atomically. A published store is not overwritten.

```python
from training_store import TrainingStore

store = TrainingStore('/path/to/nvme/document-store-fp16-v1')
records = store.read_span(document=123, start=10, length=128)
records = store.read_rows([456, 457, 1000, 456])
features = store.features(records)
store.close()
```

`read_rows` combines consecutive reads and restores requested order and repeats.
Feature arrays and presence flags are FP16; dense normalization uses higher precision
accumulation before returning FP16. Statistics are already normalized. Do not pass
them through the old raw-statistics normalizer again.

To use the store, set the experiment's `index` to `STORE/catalogue.sqlite`,
`training_store` to `{"path": "STORE"}`, and `precision` to `"float16"` on CUDA.
The existing feature configuration remains the schema/reference declaration.
Window inputs and collation preserve FP16. CUDA autocast performs FP16 compute;
master parameters, AdamW state, losses and merged inference scores stay FP32.
Dynamic gradient scaling is checkpointed and restored, including the transition
from pretraining to fine-tuning. Overflowed draws are recorded as exposure but
never counted as optimizer updates. Repeated overflow stops the run.

An explicitly configured `"allow_building": true` also permits training while
conversion finishes. Only a shard with its atomic completion marker and matching
build ID/size is readable; partial files are never used. Windows touching pending
shards use the original pipeline rounded to FP16. Completed-store and fallback
inputs are numerically identical, preserving full-corpus sampling from the start.
Workers discover newly completed shards automatically. The immutable plan, source
signatures and catalogue are checked; a build failure stops the training reader.
Each training record and ClearML report counts NVMe versus fallback pages.
Without this explicit setting an unpublished store remains unusable.

A completed store can be copied byte-for-byte to another disk. Keep the plan,
manifest, catalogue, document mapping, shard completion markers and checksums.
Verify `checksums.json` before publishing the destination. Use the destination's
catalogue in a new training configuration and disable `allow_building`.

To evaluate a checkpoint originally trained from this native store after a move:

```bash
python -m page_sequence.runner evaluate --checkpoint /runs/finetune_best.pt \
  --training-store /ssd/document-store-fp16-v1 \
  --manifest /data/dev.jsonl --split dev --device cuda --output /runs/dev.json
```

The evaluator verifies the catalogue checksum, immutable plan checksum, build ID
and feature schema. A published native store supplies all features itself, so
evaluation does not require the old LMDB locations. This override cannot convert
an older LMDB/FP32 checkpoint to a different FP16 input representation silently.

Build progress is in `status.json`, completed shard summaries in `shards/*.done.json`,
and numerical comparisons in `validation-report.json`. Reference comparisons require
exact equality with current FP32 model inputs rounded to FP16 and exact dense source
values, labels, presence and page order. The source catalogue and original LMDBs remain
untouched, including while the FP32 pilot continues running.
