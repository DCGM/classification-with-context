# W512 stream/document and tail-strategy comparison (2026-10-28)

Same r16 W512 checkpoint, weak-type weight 0.05, step 15927. All four arrangements
use identical panel rows and labels: 431,396 pages, 1,979 documents, 3,758 curated
pages. Full type probabilities are retained. Historical anchored cases were reused;
only the twenty regular-tail cases (five overlaps × two processing modes × two
metadata modes) were computed. Window length stays 512; stride is 512 minus overlap.

Old tail: final full window shifted back to end at the scan end. New tail: keep
regular stride, allow a short final window, stop once the endpoint is covered.
Fusion, center-logit merge weights, pagination and checkpoint are unchanged.

## Complete document boundaries supplied

Curated page-type weighted F1 (%).

| Overlap | Stream old | Stream new | Per-document old | Per-document new |
|---:|---:|---:|---:|---:|
| 256 | 92.109 | 92.109 | 92.341 | 92.368 |
| 128 | 92.194 | 92.194 | 92.341 | 92.345 |
| 64 | 92.191 | 92.191 | 92.341 | 92.366 |
| 32 | 91.991 | 91.991 | 92.341 | 92.372 |
| 16 | 92.181 | 92.181 | 92.341 | 92.364 |

## Document boundaries unknown; true scan borders retained

Curated page-type weighted F1 (%).

| Overlap | Stream old | Stream new | Per-document old | Per-document new |
|---:|---:|---:|---:|---:|
| 256 | 91.272 | 91.272 | 92.318 | 92.314 |
| 128 | 91.328 | 91.328 | 92.318 | 92.323 |
| 64 | 91.618 | 91.618 | 92.318 | 92.343 |
| 32 | 91.199 | 91.199 | 92.318 | 92.348 |
| 16 | 91.092 | 91.092 | 92.318 | 92.315 |

## Interpretation

Regular tails preserve all curated continuous-stream predictions: the scan only
changes at its final window. Per-document F1 changes are below 0.04 percentage
points across these settings. The change is too small to rank overlaps reliably.
Most documents fit in one window; the 151 documents longer than 512 pages contain
229 curated pages. About 150 documents have changed tail layouts. Only 2–3 curated
page-type predictions change per per-document setting when switching tail policy.

In per-document mode, old-tail false internal splits are 42 without supplied
metadata and 17 with it. Under the new tail they are respectively 42/17 at
overlap 256, 42/16 at 128, 42/16 at 64, 41/16 at 32 and 43/16 at 16.
All per-document settings retain one front/back error. Each mode excludes first-page
pairs. There are no positive pair-boundary targets in that arrangement, so
boundary F1/recall is not informative. Raw network predictions are scored; no
boundary-copying override is applied.

## Processing cost with supplied metadata

Time is a single measurement including feature reads and inference; cache and
scheduling noise limit interpretation. Page-position counts are deterministic.

| Overlap | Old document positions | New document positions | New document seconds |
|---:|---:|---:|---:|
| 256 | 508,858 | 486,436 | 21.91 |
| 128 | 490,938 | 454,436 | 21.26 |
| 64 | 486,330 | 442,340 | 21.31 |
| 32 | 484,794 | 436,772 | 21.42 |
| 16 | 483,770 | 434,052 | 20.88 |

## Artifacts and provenance

Combined machine-readable report: `/home/ikohut/tmp/tail-strategy-comparison-r16-w512/summary.json`.
Source directories:

- continuous, anchored: `/home/ikohut/tmp/overlap-comparison-r16-w512-weak005`
- continuous, regular: `/home/ikohut/tmp/overlap-comparison-r16-w512-weak005-regular`
- per_document, anchored: `/home/ikohut/tmp/overlap-comparison-r16-w512-weak005-per-document`
- per_document, regular: `/home/ikohut/tmp/overlap-comparison-r16-w512-weak005-per-document-regular`

Checkpoint, catalogue and manifest hashes match across all source reports; panel
rows, document spans, labels and annotation flags were checked for exact equality.
Historical code hashes differ because reports predate the tail-policy option.
Each report records its own sweep-code hash and per-case inference configuration.
