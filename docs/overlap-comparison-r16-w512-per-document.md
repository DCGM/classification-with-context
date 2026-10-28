# W512 per-document overlap comparison (2026-10-28)

Same checkpoint and exact panel as [the continuous sweep](overlap-comparison-r16-w512.md):
r16 W512 weak-type 0.05, step 15927; 431,396 pages in 1,979 documents, 3,758 curated labels.
Each document is a separate scan with true scan-start/end flags. Complete document
metadata is supplied or unknown. Merge weights and all other inputs remain unchanged.

| Overlap | Type F1 unknown | Type F1 supplied | Processed positions | Reduction | Supplied seconds |
|---:|---:|---:|---:|---:|---:|
| 256 | 92.32% | 92.34% | 508,858 | 0.00% | 22.36 |
| 128 | 92.32% | 92.34% | 490,938 | 3.52% | 21.34 |
| 64 | 92.32% | 92.34% | 486,330 | 4.43% | 21.57 |
| 32 | 92.32% | 92.34% | 484,794 | 4.73% | 21.64 |
| 16 | 92.32% | 92.34% | 483,770 | 4.93% | 22.41 |

Every overlap produces exactly the same class predictions on all 3,758 curated
pages in each metadata mode. All overlaps retain one front/back error. Side F1
is effectively unchanged (94.98%). Internal false boundary predictions remain
42 without document metadata and 17 with it, among 429,417 internal pairs.
Boundary recall/F1 is uninformative here: every true document start is excluded
as the first page of a separate scan.

Only 151 documents (109,930 pages; 229 curated labels) exceed 512 pages. Because
the final window is anchored to the document end, changing stride actually changes
window layouts in only 48 documents containing 73 curated labels. Documents needing
only the first and final windows can retain identical layouts despite changing stride.
Thus identical curated scores provide limited evidence about very long documents.

At overlap 64 the supplied-mode weak DB agreement F1 is 90.6671%, versus 90.6668%
at overlap 256; 129 predictions differ across all pages, with 26 fixes and 24
regressions against weak labels. This is effectively unchanged and weak labels
are not verified truth.

No substantial per-document throughput gain: overlap 64 reduces processed positions
by 4.43%, with observed times 22.36 to 21.57 seconds. Single-pass timings have cache
and scheduling noise; overlap 16 took 22.41 seconds despite 4.93% fewer positions.
Keep the current default pending an explicit inference policy choice; the strong
continuous-stream efficiency gain does not transfer to this per-document panel.

Outputs (aligned probabilities, panel and full reports):
`/home/ikohut/tmp/overlap-comparison-r16-w512-weak005-per-document`.
Baseline F1 exactly matches the earlier per-document comparison. Checkpoint, manifest,
catalogue and sweep-code SHA256 identities are recorded in report.json.

Run with `python -m page_sequence.compare_overlaps --processing per_document`;
five targeted evaluator tests pass.
