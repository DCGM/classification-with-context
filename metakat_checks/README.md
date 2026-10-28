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
- `metakat.json`: prediction input, with scan order, mappings and image/ALTO dimensions;
- `metakat.with-page-types.json`: an additional input with fixed training page-type
  targets supplied as annotated value objects;
- `manifest.json`: native row IDs, public document identities, ordering evidence,
  source paths, checksums, DB labels, human labels and true document starts/ends;
- `READY.json`: written last, once copying and basic content checks succeed.

In the original `metakat.json`, only DB pagination is seeded into model input. Every number is `annotated: true`,
including an explicit empty string when the DB has no usable entry, so the number
engine cannot invent a different signal in a controlled parity run. Original
brackets and other raw numbering syntax are retained. Reference page types,
sides and boundaries are never injected into `metakat.json`; they remain in the
manifest. Classifier predictions are therefore not bypassed.

### Inputs with supplied page types

`metakat.with-page-types.json` adds the effective training targets to a copy of
`metakat.json`. It follows `page_sequence.schema.Page.target()` exactly: a curated
label overrides the DB label, otherwise the DB label is used. Ignored/missing DB
labels leave `pageType` unset; unsupported curated labels fail rather than falling
back to DB. Labels are converted to MetaKat's NDK spelling, including
`FlyLeaf` → `flyleaf`. Each supplied value has a deterministic ID,
`confidence: 1.0` and `annotated: true`. This confidence is a fixed external-input
score, not a calibrated model probability or the weak-label loss coefficient.
DB values remain weak labels in their provenance; the flag does not assert human
verification.

| Set | Supplied types | Human overrides | DB targets | Unset types | Fully supplied batches |
|---|---:|---:|---:|---:|---:|
| Consistency | 5,063 | 6 | 5,057 | 278 | 44 |
| Evaluation | 3,451 | 115 | 3,336 | 2,665 | 25 |

All other inputs, including pagination, scan order and geometry, stay identical.
Document boundaries and sides remain unset. The original `metakat.json` remains
the input for prediction-parity runs. The supplied-type variant instead tests
annotation preservation and downstream use of fixed types: the binder skips the
core for fully supplied batches and only writes missing types for partial batches
while still processing their complete context.

To regenerate or verify these variants with the MetaKat environment:

```bash
/home/ikohut/python_env/metakat/bin/python -m metakat_checks.supply_page_types
/home/ikohut/python_env/metakat/bin/python -m metakat_checks.supply_page_types --check-only
```

`supplied-page-types.json` records per-batch target provenance counts and input
checksums; `supplied-page-types.validation.json` records a separate verification.
Generation and verification both use the MetaKat schema and CLI input loader.

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
