# W512 overlap comparison (2026-10-28)

Checkpoint: r16 W512, optional document metadata, fine-tuning weak-type weight 0.05,
step 15927. Continuous manifest-order stream: 431,396 pages, 1,979 complete documents,
3,758 curated labels. Same pages, weights, pagination and center-logit merger in every pass.
Only stride changes: 512 minus overlap. Both complete document metadata supplied and unknown.

| Overlap | Type F1 unknown | Type F1 supplied | Boundary F1 unknown | Boundary F1 supplied | Processed positions | Supplied-mode seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 91.27% | 92.11% | 91.83% | 99.32% | 862,720 | 16.85 |
| 128 | 91.33% | 92.19% | 91.94% | 99.32% | 575,488 | 12.12 |
| 64 | 91.62% | 92.19% | 91.60% | 99.32% | 493,056 | 11.05 |
| 32 | 91.20% | 91.99% | 91.80% | 99.40% | 460,288 | 9.65 |
| 16 | 91.09% | 92.18% | 91.55% | 99.35% | 445,440 | 8.97 |

64-page overlap is a promising compromise: 42.85% fewer processed positions than
256-page overlap, with slightly better curated type F1 in both metadata modes.
Unknown-metadata boundary F1 declines by 0.24 percentage points. At 64-page overlap,
curated predictions fix/regress 27/14 pages without metadata and 20/17 with metadata
relative to the baseline. Supplied-mode front/back confusion remains one page for
overlaps 256, 128, 64 and 32; overlap 16 produces two.

These are single panel/alignment results, not evidence of monotonic improvement.
Timing includes feature reads and inference but excludes output writing; cache effects
and one pass per setting limit speed comparisons. This sweep is continuous-only:
per-document processing must be compared before changing the full-DB export policy.

All aligned probabilities and the exact panel are retained at:
`/home/ikohut/tmp/overlap-comparison-r16-w512-weak005`.
The 256-overlap page-type F1 matches the earlier continuous comparison exactly.

Provenance:

- Checkpoint SHA256: `e52a49a453ce844e449d92918cf359260192e13bf9a7edb34b681cbff0ac5c4a`
- Manifest SHA256: `1a3779cb4c65c9b235122d49d69e576d39ead3d6ed5e5c768f3443e20b1f1b22`
- Catalogue SHA256: `4cc82720b3330744c92cc54102bb989502d0b9e5dfefeb8488f62a2f40afec51`
- Sweep code SHA256: `003708b4020552371f4788c0e07e10dd065a47b8354cd75d580a3e167210f225`

Reproduce with `python -m page_sequence.compare_overlaps`; see the package README.
