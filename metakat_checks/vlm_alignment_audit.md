# VLM alignment audit — 2026-10-29

Scope: the VLM cores in MetaKat's `biblio-spans` checkout. Alignment locates
already-extracted values in OCR; it neither changes the VLM prompt nor establishes
extraction accuracy. No paid requests were submitted for this audit.

## Title pages: monographs and periodicals

Both cores use `biblio_core_engine_vlm.py`. The final dataset helper is
`/home/ikohut/Projects/text-geometry-aligner/scripts/align_metadata_json.py`.
The periodical preparation README explicitly invokes this helper, additionally
skipping the `type` code key. The monograph preparation README uses the same helper.

The rationale is recorded in the aligner session
`/home/ikohut/.claude/projects/-home-ikohut-Projects-text-geometry-aligner/380169b3-6881-402a-a5e0-3746e6cbf764.jsonl`,
particularly the exchanges on 2026-10-22 at 09:43–09:54 UTC:

- Sharing across top-level keys is necessary because an agent is also named in
  a responsibility statement, or one firm is publisher and printer. Values within
  a key still compete for words. This must be opt-in, not a general default.
- Sibling proximity chooses an equally good occurrence near related values,
  such as a place near its imprint's publisher/date. It cannot exchange occupied
  matches and cannot guarantee the correct occurrence.
- Equal-quality selection prefers tighter spans. This is already an aligner
  default and does not need a second MetaKat implementation.
- Codes and inferred publishers are not printed transcriptions to locate.
- Lowercase/punctuation normalization and minimum similarity 0.5 are the
  helper's explicit settings. Fuzzy CER for long values defaults to 0.20.
- The session reports byte-identical outputs after moving this logic from the
  script into the library on its 100-page GT panel. This was a refactor check,
  not independent verification of alignment accuracy.

MetaKat previously used ordinary exact/fuzzy candidates without these optional
sharing/proximity settings and allowed inferred publisher strings to compete for
OCR words. Both cores now use the dataset's sharing, proximity, normalization,
minimum similarity and inferred-publisher filtering. Existing page-order selection
and the bounded 10-second CP-SAT limit remain. TOC entry ordering/column reordering
is inappropriate here: bibliographic JSON fields do not follow printed order.

## Book TOCs

The final helper is `books_toc.2026-10-20/align_toc_review.py` under
`/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/smart_digiline/`.
The originating dataset session is
`/home/ikohut/.codex/sessions/2026/10/23/rollout-2026-10-23T13-24-17-01a16881-28c3-7d21-a101-08bccc9c83cf.jsonl`.
Its 2026-10-24 trials established ordered group anchors with unordered fields,
then optional conservative column reordering. The final 1,713-TOC run disabled
column reordering for 24 TOCs with parallel title versions.

MetaKat now uses that conditional policy, with group depth 2, lowercase/punctuation
normalization and minimum similarity 0.5. It retains its indicated-page/next-page
scoping and bounded CP-SAT selection. Unlike the review exporter, MetaKat reads
only the first title, part number and page reference; it does not match the whole
designation or descriptions, so the exporter's `designation` word-sharing exception
has no corresponding field here. Pages remain separate, avoiding stitched-page
column-order confusion.

## Article TOCs and chapter destination pages

Their final dataset helpers are respectively:

- `articles_toc.2026-10-20/align_toc_review.py`: ordered groups, conditional columns,
  minimum similarity 0.5; `designation` and `citation` can share words with their
  component fields. Code keys including role/type are excluded.
- `chapters_title_pages.2026-10-20/scripts/align_destination_review.py`: ordered
  groups and conditional columns; inherited `pageReference` and other TOC
  locator fields are excluded from destination-image geometry. Whole designations
  can share words. Its CP-SAT limit is 60 seconds.

The destination collection/review session is
`/home/ikohut/.codex/sessions/2026/10/23/rollout-2026-10-23T23-04-43-01a16a94-9039-78b2-8ac4-225b96bc25e2.jsonl`.
It distinguishes earlier heading-layout checks used to select pages from the
later geometry of corrected VLM values prepared for Label Studio. Collection
boxes are provenance, not the final review alignment.

There are currently no separate article-TOC or destination-page VLM engines in
this MetaKat checkout. Destination analysis remains YOLO/ALTO and chapter-to-page
resolution remains its separate fuzzy alignment stage. No speculative engine or
unused configuration was added for these dataset-only workflows.
