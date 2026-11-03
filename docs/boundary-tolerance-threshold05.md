# Recall-only boundary tolerance, threshold 0.5

Extends the [ambiguous-ending evaluation](boundary-ambiguity-threshold05.md) over
its same 193 development streams, selected checkpoints and original validation
inference. All scores are retained outputs; no inference or threshold was changed.

## Definition

For every true boundary at scan position b, recall counts it as found if any
detected boundary (probability >= 0.5) occurs in [b-k, b+k], for k=0..5. Distances
are observed scan-page positions, not database row-ID differences or printed
pagination. Matching never crosses scan streams. Scan first positions remain
excluded because they do not have pair-boundary predictions.

The true boundary keeps its original reference preceding-page type: searching
nearby positions does not change its ambiguous/other/unknown category. Unknown
page types remain separate. A detected boundary can cover more than one nearby
true boundary in this requested proximity-recall diagnostic. A separate maximum
one-to-one matched count measures sensitivity to that choice.

**Exact precision, FP counts and F1 remain unchanged.** No nearby false split is
reclassified as an exact TP, and no F1 is synthesized from relaxed recall and
exact precision. The tolerant missed count means "no prediction within radius",
not a replacement confusion matrix. At radius zero, every group exactly matches
its original TP/FN. Recall is checked to be nondecreasing in every group.

## 180-stats model, document metadata unknown

| Tolerance | Ambiguous recall (missed / 967) | Other known recall (missed / 811) | Unknown recall (missed / 8) | Overall recall (missed / 1,786) |
|---|---:|---:|---:|---:|
| Exact | 81.08% (183) | 93.83% (50) | 50.00% (4) | 86.73% (237) |
| ±1 | 82.01% (174) | 94.08% (48) | 50.00% (4) | 87.35% (226) |
| ±2 | 82.73% (167) | 94.33% (46) | 62.50% (3) | 87.91% (216) |
| ±3 | 83.14% (163) | 94.33% (46) | 62.50% (3) | 88.13% (212) |
| ±4 | 83.35% (161) | 94.70% (43) | 87.50% (1) | 88.52% (205) |
| ±5 | 83.66% (158) | 94.82% (42) | 87.50% (1) | 88.75% (201) |

At ±5, 36 of 237 originally missed boundaries have some nearby detection: 25
ambiguous, eight other known and three unknown. This leaves 201 misses, including
158 ambiguous endings. Exact FP remain 76 (22 ambiguous, 51 other known, 3 unknown).
Exact overall precision remains 95.32%.

The one-to-one sensitivity count at ±5 is 1,573/1,786 (88.07% recall), versus
1,585/1,786 (88.75%) under any-nearby matching. The difference is 12 boundaries
where proximity recall can share a prediction across nearby truths.

## Overall recall across all three models, document metadata unknown

| Tolerance | Previous best | New mmBERT / 147 stats | New mmBERT / 180 stats |
|---|---:|---:|---:|
| Exact | 90.20% (175 missed) | 91.83% (146 missed) | 86.73% (237 missed) |
| ±1 | 91.10% (159 missed) | 93.00% (125 missed) | 87.35% (226 missed) |
| ±2 | 91.49% (152 missed) | 93.51% (116 missed) | 87.91% (216 missed) |
| ±3 | 91.77% (147 missed) | 93.67% (113 missed) | 88.13% (212 missed) |
| ±4 | 92.05% (142 missed) | 93.95% (108 missed) | 88.52% (205 missed) |
| ±5 | 92.27% (138 missed) | 94.18% (104 missed) | 88.75% (201 missed) |

## 180-stats ambiguous ending types

| Type | True boundaries | Exact misses | ±1 misses | ±2 misses | ±3 misses | ±4 misses | ±5 misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| CalibrationTable | 241 | 51 | 51 | 51 | 51 | 51 | 49 |
| Cover | 148 | 48 | 44 | 44 | 43 | 43 | 43 |
| FlyLeaf | 168 | 40 | 38 | 35 | 34 | 34 | 34 |
| Spine | 157 | 27 | 26 | 24 | 23 | 23 | 22 |
| Jacket | 238 | 10 | 8 | 6 | 6 | 6 | 6 |
| Blank | 15 | 7 | 7 | 7 | 6 | 4 | 4 |

Only two of the 51 misses after CalibrationTable pages have a detected boundary
within ±5; none have one within ±4. Generic Cover misses fall from 48 to 43,
FlyLeaf from 40 to 34 and Spine from 27 to 22. Thus most ambiguous-ending misses
are not explained by a small location offset at the fixed 0.5 threshold.

## With complete supplied document metadata

Recall does not change at any radius: previous and new-147 remain 99.89% overall
(two misses), and new-180 remains 99.94% (one miss). Other-known-type recall is
100% throughout. Exact FP/precision remain as in the original evaluation.

## Nearest detected boundary offsets for 180 stats

Offsets below are restricted to ±5, and select the earlier prediction on an
equal-distance tie. Negative means a cut before the true document start.

| Offset | All true boundaries | Ambiguous-ending true boundaries |
|---|---:|---:|
| -5 | 1 | 1 |
| -4 | 5 | 2 |
| -3 | 1 | 1 |
| -2 | 8 | 7 |
| -1 | 8 | 7 |
| +0 | 1549 | 784 |
| +1 | 3 | 2 |
| +2 | 2 | 0 |
| +3 | 3 | 3 |
| +4 | 2 | 0 |
| +5 | 3 | 2 |

## Reproduction

```bash
python -m page_sequence.boundary_ambiguity \
  --root /home/ikohut/tmp/boundary-thresholds-2026-11-03 \
  --catalogue /home/ikohut/tmp/document-store-ocr-lines-sparse-v2/catalogue.sqlite \
  --output /home/ikohut/tmp/boundary-thresholds-2026-11-03/ambiguity-tolerance-threshold05.json \
  --threshold 0.5 --max-tolerance 5
```

The JSON retains all models, modes, groups, observed preceding-page types, offset
histograms, one-to-one sensitivity counts and unchanged exact metrics. A flat
group table is `ambiguity-tolerance-threshold05.csv` alongside it. Tests cover
early/late offsets, per-scan isolation, unchanged false positives and the distinction
between shared-prediction proximity recall and one-to-one matching.
