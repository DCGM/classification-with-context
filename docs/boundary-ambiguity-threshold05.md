# Boundary evaluation by ambiguous preceding page types (threshold 0.5)

This partitions the same 431,203 eligible pairs in 193 development streams,
including 1,786 true document boundaries. It uses the retained selected-checkpoint
scores from the [threshold evaluation](boundary-thresholds-2026-11-03.md). No
inference, model or decision threshold was changed.

## Definition

The preceding page determines the category: the final page of the earlier document
for a true boundary, or the preceding within-document page for a negative pair.
Thus false positives have a valid category and denominator too. The following
reference labels define ambiguous scan-edge material:

`CalibrationTable`, `Cover`, `Jacket`, `FlyLeaf`, `Blank`, `Spine`, `Edge`, `FragmentsOfBookbinding`.

All other supported page types form the **other known types** group. Explicit
front/back labels stay in that group. This is an explicit operational partition,
not a claim that every other type is inherently unambiguous. Missing/unsupported
labels remain **unknown**, rather than being silently treated as clear endings.
The reference type uses human annotation when present, otherwise the DB type;
it never uses model predictions. Of the true boundary predecessors, 807 have
human reference labels and 979 use DB labels. No OCR/private page text is read.

## 180-stats model, without supplied document metadata

| Preceding page category | True boundaries | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ambiguous | 967 | 784 | 22 | 183 | 97.27% | 81.08% | 88.44% |
| Other known types | 811 | 761 | 51 | 50 | 93.72% | 93.83% | 93.78% |
| Unknown type | 8 | 4 | 3 | 4 | 57.14% | 50.00% | 53.33% |
| All | 1786 | 1549 | 76 | 237 | 95.32% | 86.73% | 90.82% |

Ambiguous endings contain 54.1% of true boundaries but account for 183/237
(77.2%) of misses. Their miss rate is 18.9%, versus 6.2% for the other known
labels. This association supports the scan-edge ambiguity concern without proving
it is the only cause of the errors.

## Comparison across models, without supplied document metadata

| Model | Ambiguous FP / FN | Ambiguous recall | Ambiguous F1 | Other known FP / FN | Other known recall | Other known F1 |
|---|---:|---:|---:|---:|---:|---:|
| Previous best, old mmBERT / 147 stats | 22 / 137 | 85.83% | 91.26% | 88 / 34 | 95.81% | 92.72% |
| New mmBERT / 147 stats | 66 / 113 | 88.31% | 90.51% | 137 / 29 | 96.42% | 90.40% |
| New mmBERT / 180 stats | 22 / 183 | 81.08% | 88.44% | 51 / 50 | 93.83% | 93.78% |

The 180-stats model improves other-known-type F1 relative to the previous best
(93.78% versus 92.72%), with fewer false splits but more misses. For ambiguous
endings, F1 is lower (88.44% versus 91.26%). Precision/F1 between the two categories
also reflect their different boundary prevalence: 967 positives among 11,328
ambiguous-predecessor pairs versus 811 among 311,477 other-known-predecessor pairs.
Recall/miss rate provides the direct comparison at actual document endings.

## 180-stats model: ending types with true boundaries

| Reference preceding-page type | Category | True boundaries | Missed | Miss rate | FP after this type |
|---|---|---:|---:|---:|---:|
| CalibrationTable | Ambiguous | 241 | 51 | 21.16% | 6 |
| Cover | Ambiguous | 148 | 48 | 32.43% | 2 |
| FlyLeaf | Ambiguous | 168 | 40 | 23.81% | 2 |
| BackCover | Other known types | 687 | 32 | 4.66% | 10 |
| Spine | Ambiguous | 157 | 27 | 17.20% | 0 |
| Jacket | Ambiguous | 238 | 10 | 4.20% | 11 |
| Blank | Ambiguous | 15 | 7 | 46.67% | 1 |
| Map | Other known types | 44 | 6 | 13.64% | 4 |
| NormalPage | Other known types | 55 | 6 | 10.91% | 15 |
| (missing) | Unknown type | 8 | 4 | 50.00% | 3 |
| SheetMusic | Other known types | 3 | 3 | 100.00% | 1 |
| BackEndSheet | Other known types | 7 | 2 | 28.57% | 7 |
| FrontEndSheet | Other known types | 1 | 1 | 100.00% | 1 |
| Advertisement | Other known types | 1 | 0 | 0.00% | 0 |
| FrontEndPaper | Other known types | 1 | 0 | 0.00% | 0 |
| Illustration | Other known types | 1 | 0 | 0.00% | 0 |
| Imprimatur | Other known types | 1 | 0 | 0.00% | 0 |
| Index | Other known types | 3 | 0 | 0.00% | 2 |
| Table | Other known types | 4 | 0 | 0.00% | 0 |
| TableOfContents | Other known types | 3 | 0 | 0.00% | 2 |

Generic jackets are included in the ambiguous group but perform well: 10/238
missed, versus 48/148 generic covers. Blank endings have only 15 examples. Edge
and FragmentsOfBookbinding have no true-boundary predecessors in this panel.

## With complete document metadata supplied

| Model | Ambiguous FP / FN | Other known FP / FN | Unknown-type FP / FN |
|---|---:|---:|---:|
| Previous best, old mmBERT / 147 stats | 8 / 2 | 14 / 0 | 0 / 0 |
| New mmBERT / 147 stats | 10 / 2 | 23 / 0 | 1 / 0 |
| New mmBERT / 180 stats | 4 / 1 | 6 / 0 | 0 / 0 |

## Reproduction

```bash
python -m page_sequence.boundary_ambiguity \
  --root /home/ikohut/tmp/boundary-thresholds-2026-11-03 \
  --catalogue /home/ikohut/tmp/document-store-ocr-lines-sparse-v2/catalogue.sqlite \
  --output /home/ikohut/tmp/boundary-thresholds-2026-11-03/ambiguity-threshold05.json \
  --threshold 0.5
```

Full metrics and every observed preceding type are retained in the JSON; the
group-level flat table is `ambiguity-threshold05.csv` beside it. Every model and
both metadata modes exactly reconcile with the original validation TP/FP/FN.
Two tests cover direction-specific/unknown labels and assignment of negative
pairs to the same categories used for true boundaries.

## Nearby-boundary recall

The [recall-only ±1..±5 extension](boundary-tolerance-threshold05.md) retains
the exact precision/FP metrics and checks for nearby predictions within each scan.
