# R19 missed boundaries by document type

Uses the selected 180-stat checkpoint, step 16177, and the exact same 193 development
streams and boundary scores as the [threshold sweep](boundary-thresholds-2026-11-03.md).
Ground truth is bottom-level catalogue document identity. Categories come from the
cached source DB record type and parent hierarchy; no OCR, page text or live DB
queries are used. Counts reconcile exactly with the original validation at 0.5.

The panel contains 1,979 documents: 1,477 single monographs, 112 parts of multipart
works, 126 periodical issues, 187 sheet-music documents, 60 graphics and 17 components
of bound-with collections. The last 17 have DB type `monographunit` but parent type
`convolute`, so they are not counted as multipart works. All 112 multipart parts have
parent type `monographmulti`. All document record types were present in the cache.

## Complete type-pair breakdown, unknown boundaries

Pairs are unordered; either left/right direction contributes to the same row.

| Document types | True boundaries | Missed at 0.5 | Miss rate | Missed at ≈0.6373 |
|---|---:|---:|---:|---:|
| Single monograph ↔ Single monograph | 990 | 92 | 9.3% | 110 |
| Sheet music ↔ Single monograph | 242 | 37 | 15.3% | 44 |
| Periodical issue ↔ Single monograph | 154 | 16 | 10.4% | 16 |
| Multipart part ↔ Single monograph | 152 | 22 | 14.5% | 23 |
| Graphic ↔ Single monograph | 98 | 15 | 15.3% | 18 |
| Bound-with component ↔ Single monograph | 27 | 6 | 22.2% | 6 |
| Periodical issue ↔ Sheet music | 25 | 0 | 0.0% | 0 |
| Sheet music ↔ Sheet music | 24 | 22 | 91.7% | 22 |
| Multipart part ↔ Sheet music | 20 | 8 | 40.0% | 8 |
| Periodical issue ↔ Periodical issue | 18 | 6 | 33.3% | 6 |
| Multipart part ↔ Periodical issue | 13 | 1 | 7.7% | 1 |
| Graphic ↔ Sheet music | 9 | 9 | 100.0% | 9 |
| Graphic ↔ Periodical issue | 5 | 0 | 0.0% | 0 |
| Multipart part ↔ Multipart part | 5 | 1 | 20.0% | 1 |
| Bound-with component ↔ Sheet music | 1 | 1 | 100.0% | 1 |
| Graphic ↔ Multipart part | 1 | 0 | 0.0% | 0 |
| Bound-with component ↔ Periodical issue | 1 | 1 | 100.0% | 1 |
| Bound-with component ↔ Multipart part | 1 | 0 | 0.0% | 0 |

## Related-document boundaries

- Issues of the same periodical volume: 5 misses out of 9 boundaries (55.6%).
- Issues from different periodical titles: 1 miss out of 9 (11.1%).
- There are no same-title cross-volume joins in this panel.
- There are no joins between parts of the same multipart work.
- The five multipart-part ↔ multipart-part joins are between different works; one is missed.

The small issue-pair sample and absent same-work multipart joins mean the global
boundary F1 is not sufficient to characterize these difficult intended use cases.

## Grouped summary

| Pair category | True boundaries | Missed at 0.5 | Missed at ≈0.6373 |
|---|---:|---:|---:|
| Single monograph ↔ single monograph | 990 | 92 | 110 |
| Parts of different multipart works | 5 | 1 | 1 |
| Periodical issues, same volume | 9 | 5 | 5 |
| Periodical issues, different titles | 9 | 1 | 1 |
| Sheet music ↔ sheet music | 24 | 22 | 22 |
| Mixed types | 749 | 116 | 127 |
| **Total** | **1,786** | **237** | **266** |

Raising the threshold from 0.5 to the fitted precision-biased 0.63725656 adds 29
misses: 18 single-monograph pairs, seven monograph/sheet-music pairs, three
monograph/graphic pairs and one monograph/multipart-part pair. The issue-pair
misses and multipart/multipart misses do not increase over this interval.

With complete document metadata supplied, the only miss at 0.5 is a single-monograph
pair (score 0.36665913). At approximately 0.6373 a graphic/sheet-music pair is
also missed, giving two misses.

## Reproduction

`tools/analyze_boundary_document_types.py` reads the retained per-stream scores,
`catalogue.sqlite` and `/home/ikohut/tmp/page-sequence-data/hierarchy.sqlite`.
Outputs under `/home/ikohut/tmp/boundary-thresholds-2026-11-03/`:

- `document-types.json`: complete type-pair and relationship counts at four thresholds.
- `true-boundaries-by-document-type.csv`: all 1,786 boundaries with probabilities,
  preceding/current row IDs, document IDs, types and relationship categories.
- `document-type-identities.json`: DB record/parent types and hierarchy identities.

All output contains metadata and probabilities, no OCR or private page text.
