# Proposed document-oriented training store

Status: converter and standalone reader implemented in `training_store/`. The real-data
correctness pilot converted 41,494 pages and passed 4,096 reference comparisons, including
exact equality with current features rounded to FP16. Long-span and overlap reads also passed.
The full NVMe build was launched on 2026-10-25 at
`/home/ikohut/tmp/document-store-fp16-v1`, with a frozen code snapshot and one conversion worker.
No training-store migration has been performed; existing training continues. See
`training_store/README.md` for the format and resume commands. Build status/logs are under
`/home/ikohut/tmp/training-store-build/` and the destination's `status.json`.
The running FP32 experiment is a pilot. The next experiment must support native FP16
feature tensors and model computation; the current runner does not implement this yet.

## Reason and scale

The full-corpus run measured about 7,500 page presentations/s with eight workers. A live sample
showed approximately 47,000 small SSD reads/s, 182 MiB/s, and essentially continuous SSD activity.
Loading/sampling occupied about 97% of batch time. Earlier public-subset benchmarks benefited
from a smaller working set. Merely combining current UUID-keyed stores would not establish
document locality; the new physical layout must place consecutive document pages together.

The retained index has 45,099,935 pages in 1,087,561 documents:

| Partition | Pages |
|---|---:|
| Fit | 37,911,505 |
| Development | 6,700,698 |
| Test | 487,732 |

There are 624,616 documents of at most eight pages, and the largest retained document has
2,649 pages. A synthetic training window can span multiple documents, so the objective is one
contiguous read per document fragment, with optional coalescing of nearby fragments. One read
for every arbitrary synthetic batch is not guaranteed.

## Recommended baseline

Use one immutable dataset directory with large binary shards and a metadata catalogue.
Start with uncompressed, fixed-size page records readable through NumPy and bounded `pread`
calls; benchmark read-only memory mapping as an alternative. This avoids per-page key lookups,
multiple modality stores, sparse-stat decoding, and grouping/normalizing statistics each epoch.
NumPy supports accessing slices of large mapped files without loading the entire file into RAM:
https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

Illustrative layout:

```text
document-store-v1/
  manifest.json
  catalogue.sqlite
  documents.npy
  shards/
    000000.bin
    000001.bin
    ...
  checksums.json
  validation-report.json
```

This is one logical source; it need not be one enormous physical file. Begin with roughly
1–4 GiB shards. Keep each document contiguous and wholly inside one shard where possible.
Order documents by verified library/periodical/volume identity, then document identity, to
place likely hard-boundary neighbors near each other. Preserve original page-index row IDs
and record the mapping to physical locations; physical ordering must not change sampling.

Each page record contains:

| Field | Storage |
|---|---|
| Original DINOv2 vector | 768 float16 values |
| Original mmBERT vector | 768 float16 values |
| All grouped, normalized statistics | 147 float16 values |
| Original index row ID | uint64 |
| Document ID | uint32 |
| Original page order | checked int32 |
| Weak type, curated type, placement | three signed bytes, missing sentinel |
| Modality availability | one bitfield byte |

This is 3,386 bytes/page (packed, without implicit padding), about **142.2 GiB** for all retained
pages. A 128-page fragment is about 423 KiB. Document-start alignment, catalogue, manifests and build headroom add space;
target **200 GiB free** on the destination. The existing metadata catalogue is about 10.7 GiB
and can initially be retained unchanged to minimize implementation changes.

Dense vectors keep their original float16 bytes. All grouped statistics are also stored in
float16, as requested. Compute grouping, normalization, coordinate clipping and block scaling
with the current FP32 pipeline during conversion, then round the final statistics to FP16.
This adds a rounding step relative to the current reader and must be quantified in the pilot.
The next reader must deliver FP16 feature tensors directly, including dense vectors after
normalization and presence flags, without unconditionally expanding the stored dataset to
FP32. Numerically sensitive normalization reductions may use FP32 accumulation before
returning FP16 tensors. Labels, masks and identifiers retain their integer/boolean types.
The manifest records names, dimensions, byte order, source checkpoints, normalization and
schema hashes. Missing-vector flags remain explicit.

Optionally retain an additional 147 unnormalized grouped float16 coordinates per page if
normalization experiments must be possible without a rebuild. This adds about **12.35 GiB**;
check representable ranges before casting, since grouped values can exceed the raw FP16 values.
Preserve the original sparse 3,324-coordinate statistics LMDBs on Kolosus for future grouping
or feature-selection changes; normalized grouped statistics alone cannot reconstruct them.
Additional embedding models become additional versioned feature groups when needed.

## Metadata and sampling

Keep the full document hierarchy, split assignments, copy groups, original UUID/path mappings,
annotations and quarantine audits in the catalogue. A compact document offset table contains
logical first row, length, shard and byte offset. It is small enough to load independently
of the page catalogue. UUID strings and paths are not repeated in hot feature records.

Expose `read_span(document, offset, length)` and `read_rows(original_rows)` APIs. The latter
groups contiguous document fragments, deduplicates repeated reads, optionally merges nearby
byte ranges, and restores exact requested order and repeated occurrences. Workers share the
OS file cache; start without an additional large private cache per worker.

The sampler, quotas, class weighting, masks and overlap-merging logic remain the same. Changing
the physical I/O order must not change the logical batch or RNG trajectory. Document starts
are computed from adjacent sampled document IDs, not from a stored first-page classification.
Known document IDs, page order and split fields remain metadata rather than model inputs.

Initial integration can keep the existing SQLite sampler and replace only feature reads.
Then move worker-side label/mask materialization to packed metadata to remove repeated page
queries. Profile the parent sampler after I/O improves before proposing further changes.

## Alternatives to benchmark

The flat binary layout is the reference. Compression is a measured alternative, particularly
if the destination remains bandwidth-limited after fixing locality. Compare fixed blocks of
64/128/256 pages with light compression, bounded decoding buffers and no whole-document
decompression for a short crop. Zarr sharding can provide independently readable chunks inside
larger storage objects, but chunk size and read amplification need measurement:
https://zarr.readthedocs.io/en/latest/user-guide/performance/

A document-block LMDB is another option if a key/value interface is preferred. Use one value
per block containing all modalities, rather than separate values per page/modality. LMDB's
overflow pages and allocation overhead should be measured against flat files:
https://lmdb.readthedocs.io/en/release/

Do not choose a compressed format solely from a small warm-cache speed result.

## Build and migration sequence

1. Inventory immutable input LMDBs and metadata already available on Kolosus. The six current
   vector stores occupy about 199 GiB locally. Verify/reuse existing remote copies where
   possible; archive missing inputs before any later deletion. OneDrive backup is a separate
   space-recovery operation and its local removal requires the user's instruction.
2. Freeze the input index, label vocabulary, normalization and source signatures. Retain all
   existing exclusions and held-out partitions. Conversion reads vectors and metadata only;
   source private text/OCR is never required.
3. Build a representative pilot on NVMe with a fixed memory budget, bounded page chunks,
   per-shard checksums and resumable progress. Include public/private pages, annotated and
   rare-type anchors, short documents, long books, and same-volume boundaries.
4. If sourcing from Kolosus, avoid individual LMDB key lookups across the network. Prefer a
   converter running near the remote source disks that produces completed shards, followed
   by sequential shard transfers to NVMe. Alternatively stream large converted blocks to
   the local writer. Local conversion from the SATA sources can compete with the ongoing
   training reads, so measure/throttle it rather than assuming NVMe output removes contention.
   On this host `/mnt/kolosus` is a locally mounted ZFS pool, not an NFS mount. Benchmark
   sequential LMDB cursor scans against document-order lookups on that pool. A promising
   conversion route is one cursor pass per modality, a disk-backed key-to-output-offset
   index, and placement into preallocated NVMe records. That trades potentially slow random
   source reads for random writes on NVMe; it must handle one source key mapping to multiple
   retained rows and reproduce the current public/private lookup precedence exactly.
   A small 768-page source comparison during the running experiment measured about 1,046
   pages/s from SATA versus 34 pages/s from Kolosus. The first converter therefore uses the
   existing SATA vector stores with one worker and bounded buffers; this shares I/O with
   the current experiment. Kolosus remains the archive. These small measurements select a
   conversion source only; they do not establish cold-cache training throughput.
5. Validate every emitted record structurally and compare a stratified sample with the
   existing reader. Preserve original dense bytes and compare labels, availability and masks
   exactly. Compare stored statistics exactly against the current pipeline rounded to FP16,
   and report rounding error against its FP32 output. Replay identical seeded sampler windows
   and compare losses and merged inference outputs with documented precision tolerances.
   Before launching the next full experiment, implement and verify native FP16 model/input
   handling, finite forward/backward/optimizer updates, validation and checkpoint resume.
   Record any FP32 numerical accumulations or optimizer state explicitly; FP16 storage alone
   does not establish FP16 training support. Keep the running FP32 pilot unchanged.
6. Benchmark the existing source and pilot with identical replayed windows, page budgets,
   batch 64/window 128, and eight workers. Report end-to-end throughput, physical I/O,
   memory, wait time and read amplification. Include a working set exceeding available RAM,
   repeated alternating runs, and long-stream validation. Record cache limitations explicitly.
7. Choose the measured layout, build all shards without a corpus-sized RAM array or duplicate
   full staging dataset, and publish the manifest atomically after validation. Keep source
   archives and a rollback path. Do not delete sources being used by the current experiment.
8. Use the store for subsequent runs, or make an explicit checkpoint migration after proving
   semantic equivalence. Existing strict provenance checks reject a source/code
   swap. A controlled migration must preserve optimizer, sampler and RNG state and record
   the format change and equivalence checks in provenance.

Implement the pilot as a separate package/tool until integration is ready: adding modules to
`page_sequence` changes the running experiment's source fingerprint on a later resume.
Also benchmark parallel validation loading; its current synchronous reading is a separate
source of wall-clock overhead.
