# Exploratory boundary threshold sweep (2026-11-03)

All three selected fine-tuning checkpoints were evaluated with their original frozen
validation inference settings on the same 193 development streams: 431,203 eligible
adjacent-page positions, 1,786 true boundaries and 429,417 within-document positions.
The first page of each stream is excluded, as in training validation. Ground truth
is catalogue bottom-level document identity. This is an exploratory fit on the same
development panel, not a held-out estimate. No new split was introduced.

Unknown and supplied metadata modes were collected independently. The latter supplies
complete document start/end flags to every page. Predictions remain raw model outputs;
no annotation-copying overrides, neighborhood suppression or segmentation resolver
is applied. Scan endpoints, pagination and overlapping-window merging remain unchanged.

The score collector caches at most 64 MiB of features per stream between modes.
Every model and both modes exactly reproduce the saved validation TP/FP/FN counts
at threshold 0.5. Scores are retained so further operating points need no inference.

Thresholds below are rounded for readability; JSON/CSV retain exact values. Decisions
use `probability >= threshold`; tied scores are never split. The F1 optimum is the
exact maximum over all distinct scores, choosing the highest threshold on a tie.
Missed-boundary budgets choose the highest threshold meeting the budget. False-split
budgets maximize recall, then choose the highest threshold achieving it.

## F1 maxima without document metadata

| Model | Step | F1 at 0.5 | Best F1 | Threshold ≈ | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Old mmBERT, 147 stats (r16/r17) | 15927 | 91.56% | 91.91% | 0.582979 | 91 | 190 |
| New mmBERT, 147 stats (r18) | 15000 | 90.16% | 91.30% | 0.655556 | 131 | 176 |
| New mmBERT, 180 stats (r19) | 16177 | 90.82% | 91.20% | 0.290238 | 118 | 190 |

## Without supplied document metadata

| Model | Goal | Threshold ≈ | FP | FN | F1 |
|---|---|---:|---:|---:|---:|
| Old mmBERT, 147 stats (r16/r17) | Default | 0.5 | 122 | 175 | 91.56% |
| Old mmBERT, 147 stats (r16/r17) | Maximum F1 | 0.582979 | 91 | 190 | 91.91% |
| Old mmBERT, 147 stats (r16/r17) | At least 50% fewer FN | 0.13262 | 420 | 87 | 87.02% |
| Old mmBERT, 147 stats (r16/r17) | At least 90% fewer FN | 0.00236031 | 3,894 | 17 | 47.50% |
| Old mmBERT, 147 stats (r16/r17) | Zero FN | 1.92778e-05 | 52,394 | 0 | 6.38% |
| Old mmBERT, 147 stats (r16/r17) | At least 50% fewer FP | 0.76634 | 61 | 243 | 91.03% |
| Old mmBERT, 147 stats (r16/r17) | At least 90% fewer FP | 0.988735 | 12 | 614 | 78.92% |
| Old mmBERT, 147 stats (r16/r17) | Zero FP | 0.999936 | 0 | 1,625 | 16.54% |
| New mmBERT, 147 stats (r18) | Default | 0.5 | 212 | 146 | 90.16% |
| New mmBERT, 147 stats (r18) | Maximum F1 | 0.655556 | 131 | 176 | 91.30% |
| New mmBERT, 147 stats (r18) | At least 50% fewer FN | 0.0986307 | 829 | 73 | 79.16% |
| New mmBERT, 147 stats (r18) | At least 90% fewer FN | 0.00329865 | 4,754 | 14 | 42.64% |
| New mmBERT, 147 stats (r18) | Zero FN | 0.000133127 | 23,216 | 0 | 13.33% |
| New mmBERT, 147 stats (r18) | At least 50% fewer FP | 0.752355 | 104 | 214 | 90.81% |
| New mmBERT, 147 stats (r18) | At least 90% fewer FP | 0.982123 | 21 | 513 | 82.66% |
| New mmBERT, 147 stats (r18) | Zero FP | 0.999845 | 0 | 1,512 | 26.60% |
| New mmBERT, 180 stats (r19) | Default | 0.5 | 76 | 237 | 90.82% |
| New mmBERT, 180 stats (r19) | Maximum F1 | 0.290238 | 118 | 190 | 91.20% |
| New mmBERT, 180 stats (r19) | At least 50% fewer FN | 0.0763321 | 263 | 118 | 89.75% |
| New mmBERT, 180 stats (r19) | At least 90% fewer FN | 0.000911399 | 3,023 | 23 | 53.65% |
| New mmBERT, 180 stats (r19) | Zero FN | 3.30452e-06 | 48,604 | 0 | 6.85% |
| New mmBERT, 180 stats (r19) | At least 50% fewer FP | 0.720013 | 38 | 296 | 89.92% |
| New mmBERT, 180 stats (r19) | At least 90% fewer FP | 0.970687 | 7 | 619 | 78.85% |
| New mmBERT, 180 stats (r19) | Zero FP | 0.999352 | 0 | 1,489 | 28.52% |

## With complete supplied document metadata

| Model | Goal | Threshold ≈ | FP | FN | F1 |
|---|---|---:|---:|---:|---:|
| Old mmBERT, 147 stats (r16/r17) | Default | 0.5 | 22 | 2 | 99.33% |
| Old mmBERT, 147 stats (r16/r17) | Maximum F1 | 0.815161 | 11 | 5 | 99.55% |
| Old mmBERT, 147 stats (r16/r17) | At least 50% fewer FN | 0.25128 | 36 | 1 | 98.97% |
| Old mmBERT, 147 stats (r16/r17) | At least 90% fewer FN | 0.0664435 | 100 | 0 | 97.28% |
| Old mmBERT, 147 stats (r16/r17) | Zero FN | 0.0664435 | 100 | 0 | 97.28% |
| Old mmBERT, 147 stats (r16/r17) | At least 50% fewer FP | 0.815161 | 11 | 5 | 99.55% |
| Old mmBERT, 147 stats (r16/r17) | At least 90% fewer FP | 0.979984 | 2 | 21 | 99.35% |
| Old mmBERT, 147 stats (r16/r17) | Zero FP | 0.999511 | 0 | 133 | 96.13% |
| New mmBERT, 147 stats (r18) | Default | 0.5 | 34 | 2 | 99.00% |
| New mmBERT, 147 stats (r18) | Maximum F1 | 0.840783 | 10 | 6 | 99.55% |
| New mmBERT, 147 stats (r18) | At least 50% fewer FN | 0.163676 | 87 | 1 | 97.59% |
| New mmBERT, 147 stats (r18) | At least 90% fewer FN | 0.083354 | 137 | 0 | 96.31% |
| New mmBERT, 147 stats (r18) | Zero FN | 0.083354 | 137 | 0 | 96.31% |
| New mmBERT, 147 stats (r18) | At least 50% fewer FP | 0.799442 | 13 | 4 | 99.53% |
| New mmBERT, 147 stats (r18) | At least 90% fewer FP | 0.993661 | 3 | 38 | 98.84% |
| New mmBERT, 147 stats (r18) | Zero FP | 0.998487 | 0 | 73 | 97.91% |
| New mmBERT, 180 stats (r19) | Default | 0.5 | 10 | 1 | 99.69% |
| New mmBERT, 180 stats (r19) | Maximum F1 | 0.792399 | 3 | 4 | 99.80% |
| New mmBERT, 180 stats (r19) | At least 50% fewer FN | 0.366659 | 15 | 0 | 99.58% |
| New mmBERT, 180 stats (r19) | At least 90% fewer FN | 0.366659 | 15 | 0 | 99.58% |
| New mmBERT, 180 stats (r19) | Zero FN | 0.366659 | 15 | 0 | 99.58% |
| New mmBERT, 180 stats (r19) | At least 50% fewer FP | 0.792399 | 3 | 4 | 99.80% |
| New mmBERT, 180 stats (r19) | At least 90% fewer FP | 0.971019 | 0 | 10 | 99.72% |
| New mmBERT, 180 stats (r19) | Zero FP | 0.971019 | 0 | 10 | 99.72% |

## Interpretation

Optimizing the threshold raises boundary F1 slightly for every model. The earlier
model remains best without supplied metadata: 91.91% versus 91.30% for r18 and
91.20% for r19. Their optima occur at different thresholds, so this is model-specific
calibration, not evidence of one generally optimal constant.

For r19 without metadata, reducing missed boundaries from 237 to 23 costs 3,023
false splits; eliminating misses costs 48,604. Eliminating false splits instead
misses 1,489 of 1,786 boundaries. The extreme confidence tails overlap substantially.

With complete supplied metadata, r19 is much stronger: zero misses costs 15 false
splits, or zero false splits costs 10 misses. Its maximum F1 is 99.80% (3 FP, 4 FN).

No thresholds or model configurations were changed by this analysis. Counts are
empirical on this panel; zero observed errors do not establish a guarantee on new scans.

## Reproduction and retained outputs

Collection: `tools/collect_boundary_scores.py`, launched separately with each
experiment's frozen source on `PYTHONPATH`. Analysis:

```bash
python -m page_sequence.boundary_thresholds --root /home/ikohut/tmp/boundary-thresholds-2026-11-03
```

Output directory: `/home/ikohut/tmp/boundary-thresholds-2026-11-03/`.
`threshold-report.json` retains exact operating points and metadata; `operating-points.csv`
is the flat table. Per-model directories retain paired per-stream scores, row IDs,
truth, checkpoint hashes, inference configuration and default-count verification.

Three unit tests check tied/saturated scores and compare the exact sweep against
scikit-learn precision-recall results and direct inclusive-threshold decisions.
