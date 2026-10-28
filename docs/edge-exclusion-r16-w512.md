# Artificial window-edge exclusion comparison (2026-10-28)

Same selected W512 checkpoint (step 15927, weak-type fine-tuning weight 0.05),
431,396 development pages, 1,979 complete documents and 3,758 human-labelled pages.
Compared the existing center-logit smoothing against discarding half the overlap
at both artificial window edges, followed by a 32-page linear ramp from zero.
Overlaps 256/128/64/32/16 discard 128/64/32/16/8 pages per artificial edge.
True scan edges are exempt. Regular shorter tails, inputs and checkpoint are unchanged.
Twenty exclusion cases were computed; all twenty existing baselines were reused.

At half overlap, retained regions meet without a blended region. Page outputs
and contextual vectors therefore use the retained window prediction. The boundary
pair crossing the join is retained when either endpoint is retained, preventing
a gap in pair coverage. Pairs with both endpoints discarded have zero weight.

## Complete document boundaries supplied

Human-labelled page-type weighted F1 (%).

| Overlap | Stream baseline | Stream exclusion | Document baseline | Document exclusion |
|---:|---:|---:|---:|---:|
| 256 | 92.109 | 92.078 | 92.368 | 92.368 |
| 128 | 92.194 | 92.178 | 92.345 | 92.345 |
| 64 | 92.191 | 92.132 | 92.366 | 92.366 |
| 32 | 91.991 | 92.039 | 92.372 | 92.372 |
| 16 | 92.181 | 92.181 | 92.364 | 92.364 |

## Document boundaries unknown; scan borders retained

| Overlap | Stream baseline | Stream exclusion | Document baseline | Document exclusion |
|---:|---:|---:|---:|---:|
| 256 | 91.272 | 91.321 | 92.314 | 92.314 |
| 128 | 91.328 | 91.328 | 92.323 | 92.323 |
| 64 | 91.618 | 91.512 | 92.343 | 92.343 |
| 32 | 91.199 | 91.195 | 92.348 | 92.348 |
| 16 | 91.092 | 91.064 | 92.315 | 92.315 |

## Interpretation

Per-document processing changes no human-labelled page-type predictions at any
overlap, with or without supplied boundaries. This includes all 229 labelled
pages in the 151 documents longer than 512 pages. Most documents fit one window,
and the limited labelled long-document support restricts the conclusion.
Unlabelled/DB-labelled page predictions and probabilities can still change.

False internal boundaries are also unchanged: supplied metadata gives
17/16/16/16/16 false positives at overlaps 256/128/64/32/16; unknown metadata gives
42/42/42/41/43. No positive boundary pairs exist within isolated documents,
so boundary F1 is not informative there. Per-document side weighted F1 changes
by less than 0.01 percentage points in every case.

Stream processing shows mixed, small changes. At overlap 64 with supplied
metadata, exclusion fixes one labelled prediction and regresses three:
92.191% to 92.132% weighted F1. Without supplied metadata it fixes one and
regresses five: 91.618% to 91.512%. No consistent benefit supports replacing
the current smoothing, which remains the default (`edge_ignore: 0`).

Zero disagreement under half-overlap exclusion usually means only one window
contributes; it must not be interpreted as agreement between independent windows.

## Artifacts and reproduction

Baseline reports: `overlap-comparison-r16-w512-weak005-regular` and
`overlap-comparison-r16-w512-weak005-per-document-regular` under `/home/ikohut/tmp/`.
Exclusion reports: `/home/ikohut/tmp/edge-exclusion-r16-w512-continuous/` and
`/home/ikohut/tmp/edge-exclusion-r16-w512-per_document/`.
Each retains the aligned panel and all per-page type probabilities.
Combined scores and paired changes: `/home/ikohut/tmp/edge-exclusion-r16-w512-summary.json`.
Checkpoint, catalogue, manifest and complete panel identity were verified against
each baseline. The summary records the inference-code SHA256.

Run the overlap evaluator with `--tail regular --edge-ignore-fraction 0.5` and
`--processing continuous` or `--processing per_document`. A smaller exclusion
fraction (for example 0.25) would retain a blended region and remains untested.
