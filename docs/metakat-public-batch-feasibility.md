# Storage-based evaluation candidates for MetaKat CLI batches

Initial metadata-only audit, 2026-10-28. No images, OCR, archives, prepared batches
or remote APIs were inspected in that audit. These counts use `public.256/`
storage provenance, which does **not** establish public access. They are candidate
counts only: subsequent batch preparation checks the actual DB public flag on
every selected document and page. See [the finished preparation](../metakat_checks/README.md).
MetaKat was not run during the audit. Sources: published native catalogue, `page-sequence-data/dev.jsonl` and
cached `page-sequence-data/hierarchy.sqlite`. Held-out test data were not opened.

| Document kind | Current validation panel | Entire development split |
|---|---:|---:|
| Standalone monograph | 221 | 7,008 |
| Multipart monograph unit | 39 | 824 |
| Periodical issue | 3 | 86,050 |
| Sheet music | 187 | 676 |
| Graphic | 60 | 3,538 |
| Public-storage candidate records | 510 | 105,697 |
| Pages in those records | 97,237 | 2,412,881 |

The current panel contains 1,979 documents / 431,396 pages across public and private
sources. Its public issues form only one same-volume pair plus one isolated issue.
Its 39 multipart units belong to 35 parents: two parents have three units each,
while the remaining 33 have one. This panel supports 50 batches numerically, but
cannot provide varied related-document joins without heavy reuse.

The larger held-out development split has 6,525 candidate same-volume issue groups
with at least two issues, 5,104 with at least three, and 3,634 with at least five.
It has 94 candidate multipart families with at least two units and 30 with at least
three. These are hierarchy candidates, not verified consecutive issue/part runs:
the cache records parent relationships and types, not sibling order or issue
numbers. Consecutive issue order must be verified from bibliographic metadata
before assembly. Parent `periodicalvolume` / `monographmulti` records must not be
substituted for their bottom-level issues/parts.

The public-storage development inventory has 102,476 distinct document UUIDs across
105,697 library/document records. Deduplicate copies and overlapping page UUIDs
before choosing examples. Fifteen first-page-public records were excluded because
other page provenance differed.

Proposed 50-batch mix, subject to ordering verification: ten consecutive-issue
batches, ten multipart batches, ten batches of unrelated standalone monographs,
and twenty single-document batches with varied lengths/types. Use the expanded
set for stored-feature versus MetaKat inference parity, retaining the existing
human-labelled panel as the accuracy reference. Do not assume all expanded
candidates carry human annotations. Later preparation is intended to produce
API-compatible batches and run them locally through the MetaKat CLI, not the
worker or API; scripts/results remain outside the MetaKat repository.

Detailed local audit inventories:
`/home/ikohut/tmp/metakat-public-eval-metadata/dev-panel.json` and `full-dev.json`.
