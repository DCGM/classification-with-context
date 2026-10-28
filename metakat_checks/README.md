# Local MetaKat checks

Scripts live in this repository; MetaKat is run from its own checkout and Python
environment. Batch preparation and validation do not invoke inference, the worker,
the MetaKat API, or paid services.

Prepared data root:
`/mnt/kolosus/data/metakat_worker/test_batches/page_types/`.
It contains `consistency/` and `evaluation/`, each with 50 batches:

| Set | Documents | Pages | Largest batch | Human-labelled pages |
|---|---:|---:|---:|---:|
| Consistency | 90 | 5,341 | 386 | 6 |
| Evaluation | 86 | 6,116 | 399 | 115 |

The contents use approximately 4.3 GiB and 5.6 GiB respectively.


| Prefix | Batches per set | Composition |
|---|---:|---|
| `issues_` | 10 | Two or three adjacent issues from one volume |
| `multipart_` | 10 | Two or three adjacent parts from one multipart monograph |
| `monographs_` | 10 | Two or three unrelated standalone monographs |
| `single_` | 20 | One document |

Every batch has at most 400 pages. Documents are complete and never cropped to
fit. The consistency set samples public fit documents without requiring human
labels. Evaluation uses the entire dev split, preferring documents with curated
labels rather than spending its budget mostly on unlabelled documents. Seed:
20261028. Copies and overlapping page UUIDs are excluded within and across sets.
Related-document order is verified against the read-only DB's sibling list;
UUID order is never treated as issue/part order.

`public.256` is storage provenance, not an accessibility guarantee. Preparation
requires the DB public flag on every document and page and reads assets only from
`/mnt/matylda0/ihradis/digiknihovna_public` (read-only). Selected sequences must
match the complete native-catalogue page sequence, with no internal index gaps.
Raw DB numbers must parse to the same pagination records as the native sidecar.
No large CSV dump is loaded into memory. Metadata uses bounded SQLite caches;
image copies use four threads and stream checksums; XML is processed per page.

Each batch contains:

- `images/000001_<page UUID>.jpg` and matching `alto/000001_<page UUID>.xml`;
- `metakat.json`: CLI input, with scan order, mappings and image/ALTO dimensions;
- `manifest.json`: native row IDs, public document identities, ordering evidence,
  source paths, checksums, DB labels, human labels and true document starts/ends;
- `READY.json`: written last, once copying and basic content checks succeed.

Only DB pagination is seeded into model input. Every number is `annotated: true`,
including an explicit empty string when the DB has no usable entry, so the number
engine cannot invent a different signal in a controlled parity run. Original
brackets and other raw numbering syntax are retained. Reference page types,
sides and boundaries are never injected into `metakat.json`; they remain in the
manifest. Classifier predictions are therefore not bypassed.

The images are full-resolution archived sources for useful visual review. The
manifest separately records the native model's historical image path. Preparing
the batch does not establish numerical feature parity: that comparison must still
check preprocessing, OCR reconstruction, model revisions and feature values.
Evaluation labels are sparse; these batches complement, not replace, the full
curated development evaluation.

## Reproduce and validate

Run these commands from `/home/ikohut/Projects/classification-with-context`.

Preparation uses the classification environment and an existing read-only metadata
DB tunnel. It is resumable: stable plans and completed batches are retained.
The DB connection is taken from `DATABASE_URL` or the same local tunnel default
as the repository's OCR-fetch tools. It is not recorded in batch metadata.

```bash
/home/ikohut/python_env/classification-with-context/bin/python -m metakat_checks.prepare_batches
```

Working metadata, plan and progress:
`/home/ikohut/tmp/metakat-page-type-batches/`.

Validate with the MetaKat environment, schema and CLI input loader:

```bash
/home/ikohut/python_env/metakat/bin/python metakat_checks/validate_batches.py --checksums
```

Recheck public access and actual adjacent-sibling order with the classification
environment (metadata only):

```bash
/home/ikohut/python_env/classification-with-context/bin/python -m metakat_checks.audit_public
```

Validation loads the installed source checkout specified by `--metakat-root`;
it neither loads model weights nor runs any engines. Root `summary.json` and
`validation.json` record the finished inventory. Each set also has an `index.csv`
for selecting batches without opening their manifests.

For a later local inference run, pass the batch directory and its `metakat.json`
to `python -m metakat.process_batch`, using a reviewed engine configuration and
`--pdf-downscale` for compact interactive PDFs. Keep predictions and PDFs in a
separate results directory so prepared source batches remain unchanged.

## Completed validation

All 100 batches passed the MetaKat schema and CLI `init_io` loader, image and ALTO
content checks, and SHA256 verification. The separate read-only DB audit verified
all 176 documents and 11,457 pages as public and all 40 related-document sequences
as adjacent siblings. Three synthetic regression tests cover budget/order rules,
copy exclusion, reference-label isolation and materialized input layout. No
classification inference has been performed yet.

## Context model and merging parity

`compare_context_inference` exports the selected five-bit-boundary checkpoint
as `metakat_context_v4`, then compares training-repo inference with the MetaKat
engine in its own environment. It checks fixed compact synthetic features,
including unavailable modalities, singleton/short/full windows and 1,024-page
inputs with the regular 128-page tail, with and without complete document metadata.
All five heads are checked after merging; image/OCR encoders are mocked. This
checks architecture and scanning parity, not image/OCR preprocessing parity.

```bash
python -m metakat_checks.compare_context_inference \
  --checkpoint /path/to/finetune_best.pt \
  --metakat /path/to/MetaKat \
  --metakat-python /home/ikohut/python_env/metakat/bin/python \
  --output /home/ikohut/tmp/metakat-boundary5-r16
```

The exported `context.pt` can be used as the MetaKat core's `checkpoint_path`;
the matching DINO and mmBERT weights must still be configured separately.
The artifact stores `inference.overlap_merge` with method `center_logits`,
ramp endpoint fraction 0.75 of overlap, ignored-edge fraction 0.1875 of overlap
and `ramp_power: 3`. The ignored edge precedes the active ramp.
The default overlap is 64, resolving to ramp 48 and ignore 12.
`parity-report.json` records the settings and numerical differences.
