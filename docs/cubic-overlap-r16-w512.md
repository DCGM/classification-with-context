# Cubic overlap weighting on W512

Tested the selected r16 W512 checkpoint, step 15927, weak-type fine-tuning weight 0.05.
All 20 cases use the same 431,396 development pages, 1,979 complete documents and
3,758 human-labelled pages. The 151 documents longer than 512 contain 229 human labels.
Checkpoint, catalogue and manifest hashes and the full aligned panel match the saved baseline.

## Tested policy

For overlap O: ramp R = 3O/4; ignored pages I = R/4 = 3O/16.
Weight = clip((distance - I) / (R - I), 0, 1)^3. There is no additive floor.
Distance starts at 1 for the first included page. True scan endpoints are exempt.
All weights are normalized by their sum when averaging logits. The boundary pair uses
the smaller of its two page weights; the established zero/nonzero seam exception
preserves coverage. Window-first pair predictions are excluded.
Window size 512, regular short tail, FP16 CUDA compute, FP32 merging, batch 32,
boundary threshold 0.5. The auxiliary heads do not override the pair boundary output.
No MetaKat boundary resolver is applied.

| Overlap | Ramp | Ignore | Power |
|---:|---:|---:|---:|
| 256 | 192 | 48 | 3 |
| 128 | 96 | 24 | 3 |
| 64 | 48 | 12 | 3 |
| 32 | 24 | 6 | 3 |
| 16 | 12 | 3 | 3 |

The historical baseline has a fixed ramp 128, no ignored pages, power 1,
and the old additive 0.1 floor. Thus this compares complete merging policies;
it does not isolate polynomial power alone. Historical cases were reused, not rerun.

## Complete document metadata supplied

Human-labelled page-type weighted F1 (%).

| Overlap | Stream old | Stream cubic | Document old | Document cubic |
|---:|---:|---:|---:|---:|
| 256 | 92.109 | 92.075 | 92.368 | 92.368 |
| 128 | 92.194 | 92.178 | 92.345 | 92.345 |
| 64 | 92.191 | 92.132 | 92.366 | 92.366 |
| 32 | 91.991 | 92.039 | 92.372 | 92.372 |
| 16 | 92.181 | 92.181 | 92.364 | 92.364 |

| Overlap | Stream boundary F1 old → cubic | Document false splits old → cubic | Document side F1 old → cubic | Document human fixes / regressions |
|---:|---:|---:|---:|---:|
| 256 | 99.321 → 99.296 | 17 → 17 | 94.989 → 94.983 | 0 / 0 |
| 128 | 99.321 → 99.321 | 16 → 16 | 94.981 → 94.976 | 0 / 0 |
| 64 | 99.321 → 99.321 | 16 → 16 | 94.982 → 94.982 | 0 / 0 |
| 32 | 99.396 → 99.396 | 16 → 16 | 94.983 → 94.983 | 0 / 0 |
| 16 | 99.346 → 99.346 | 16 → 16 | 94.984 → 94.984 | 0 / 0 |

## Document metadata unknown; scan endpoints supplied

Human-labelled page-type weighted F1 (%).

| Overlap | Stream old | Stream cubic | Document old | Document cubic |
|---:|---:|---:|---:|---:|
| 256 | 91.272 | 91.360 | 92.314 | 92.314 |
| 128 | 91.328 | 91.330 | 92.323 | 92.323 |
| 64 | 91.618 | 91.512 | 92.343 | 92.343 |
| 32 | 91.199 | 91.194 | 92.348 | 92.348 |
| 16 | 91.092 | 91.063 | 92.315 | 92.315 |

| Overlap | Stream boundary F1 old → cubic | Document false splits old → cubic | Document side F1 old → cubic | Document human fixes / regressions |
|---:|---:|---:|---:|---:|
| 256 | 91.834 → 91.784 | 42 → 42 | 94.984 → 94.983 | 0 / 0 |
| 128 | 91.936 → 91.976 | 42 → 42 | 94.975 → 94.970 | 0 / 0 |
| 64 | 91.596 → 91.549 | 42 → 42 | 94.979 → 94.979 | 0 / 0 |
| 32 | 91.802 → 91.834 | 41 → 41 | 94.977 → 94.980 | 0 / 0 |
| 16 | 91.549 → 91.502 | 43 → 43 | 94.983 → 94.983 | 0 / 0 |

## Interpretation

Every per-document human-labelled type prediction is identical to its corresponding
baseline, at all overlaps and in both metadata modes, including the long-document
subset. Front/back confusion remains one error. Per-document false-split counts
are also unchanged; side agreement changes by less than 0.006 percentage points.
DB-labelled and unlabelled predictions do change (45–271 pages per case), so this
does not imply identical outputs over the full dump.

Stream results are mixed. At overlap 64, supplied-metadata type F1 changes from
92.191% to 92.132%; unknown-metadata F1 changes from 91.618% to 91.512%.
This does not demonstrate an accuracy improvement from the cubic policy, although
per-document curated accuracy is retained on this panel.

## Limits and provenance

Per-document evaluation has no positive pair-boundary targets; false internal splits
are the useful boundary diagnostic there. Side scores are agreement with DB labels.
Long-document curated support is small, and most documents need no merging.
Per-class scores, long-document scores, front/back diagnostics and paired prediction
changes are retained in the detailed summary. Timing is a single run and includes reads.

Artifacts: `/home/ikohut/tmp/overlap-ramp075-ignore025-power3-r16-w512/`.
`settings.json` records every resolved policy choice. `source/` freezes the code;
each processing directory holds its report and aligned full type probabilities.
`summary.json` verifies identities and compares the saved baseline.

Checkpoint SHA256: `e52a49a453ce844e449d92918cf359260192e13bf9a7edb34b681cbff0ac5c4a`.
Inference code SHA256: `678fce609226de5f7d37fd0a833e2bb7a15c1589898377c4cab949d71c68cc22`.
