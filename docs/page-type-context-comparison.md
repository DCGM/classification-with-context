# Page-type checkpoint and inference-context comparison

## Five-bit scan inputs: interim r15 comparison

Measured on 2026-10-27 while both r15 training jobs continued. A frozen copy of
the 256 run's best checkpoint at comparison start was used: step 15,000,
training-validation curated weighted F1 91.76896%. This is an interim checkpoint,
not the eventual final selection. Its SHA256 is
`1d9c5910580d3f4872228bed0aad04b142d04d4430ce728d42579b49875f92a8`.

The paired panel is unchanged: 431,396 pages, 1,979 complete development documents,
3,758 human-labelled pages. Both modes use window 256, stride 128, edge ramp 64,
centre-weighted logits, CUDA FP16 and batch 16, without augmentation. Each isolated
document receives scan-start/end input flags; continuous mode receives these only
at the two ends of the concatenated stream. The three document-metadata bits remain
zero in both modes, matching training. Long documents use overlapping windows.

| Mode | Curated weighted F1 | Accuracy | Macro F1 | Semantic front/back errors |
|---|---:|---:|---:|---:|
| Per document | 92.7300% | 92.8153% | 71.6068% | 1 / 1,353 |
| Continuous | 91.7191% | 91.8042% | 71.4550% | 8 / 1,353 |

Per-document inference gains 1.0109 percentage points in weighted F1 and makes
38 fewer curated errors: 74 continuous-mode errors are corrected, while 36
previously correct predictions become wrong. Another 18 pages change between
incorrect classes. Across all pages, 9,053 decisions change.

| Disjoint region | Curated pages | Per-document errors | Continuous errors |
|---|---:|---:|---:|
| Beginning | 1,293 | 93 | 111 |
| Ending | 1,990 | 117 | 143 |
| Interior | 475 | 60 | 54 |

Regions use the first/last `min(16, floor(document_length/2))` pages. The gain
occurs at both document edges, with a small interior regression. For documents
of at most 256 pages, F1 improves from 91.6755% to 93.0893%; for longer documents
it changes from 91.8700% to 91.8027%, with unchanged accuracy. Per-document
front-as-back errors are zero; back-as-front errors are one, versus one and seven
continuously. These limited counts do not currently justify a new consistency loss.

This reverses the older r10 result below and supports isolated-document inference
when document boundaries are supplied. It does not isolate the effect of the new
input flags from the other training changes, nor prove that every class improves.
The same fixed development panel and stream order are used; no test data were opened.

Artifacts, including checkpoint snapshot, aligned probabilities, region counts and
the report, are under
`/home/ikohut/tmp/page-type-context-comparison-w256-r15-check/`.
Reproduce using the command below with that directory's `best-snapshot.pt`, a new
output directory, and `--batch-size 16`; omit `--training-store` to retain the
checkpoint's SSD store. Existing five-bit inference and comparison tests: 7 passed.

### Follow-up: per-document boundary predictions

The beginning/ending errors above are page-type classification errors in edge
regions, not binary boundary errors. A separate read-only pass on the same
step-15,000 snapshot, panel and inference settings measured binary heads at 0.5.
Auxiliary logits were merged with the same page centre weights before sigmoid.

The authoritative pair head produces 41 false internal boundaries among 429,417
within-document pairs (0.00955%), affecting 33 of 1,979 documents (1.67%). The
first scan page has no valid pair prediction; outer scan endpoints are known
from the input rather than inferred by this head.

| Auxiliary head | Correct true endpoints | Missed true endpoints | False positives elsewhere |
|---|---:|---:|---:|
| Document start | 1,700 / 1,979 (85.90%) | 279 | 348 |
| Document end | 1,675 / 1,979 (84.64%) | 304 | 539 |

The scan flags are learned features, not hard output overrides. Thus the auxiliary
heads do not always recognize known scan endpoints, even though the page-type
head benefits from per-document inference. These auxiliary positives are not used
as the final split decisions. Preserving supplied outer boundaries explicitly in
an application is distinct from measuring these raw model predictions.

Diagnostic script: the artifact root's `check_boundaries.py`; output:
`comparison/boundary-report.json` and `boundary-aux-probabilities.npz`. The script
adds auxiliary logit accumulation without modifying training or production inference.

## Historical r10 comparison

Measured on 2026-10-27. Select by human-labelled development page-type weighted F1;
boundary/side performance is not part of this decision. The held-out test set was not opened.

## Checkpoint selection

All ranked runs used the same development manifest (3,758 curated page labels).
These are observed single-run validation scores, not evidence that tiny differences
between the leading checkpoints are statistically reliable. The 512 consistency run
was still active when the ranking was captured.

| Run | Best step | Stage | Window | Curated weighted F1 |
|---|---:|---|---:|---:|
| `bert-mini-fp16-nvme-b64-w256-variable-whole-dropout-seed17-r10` | 15250 | finetune | 256 | 91.6311% |
| `bert-mini-fp16-ssd-b64-w256-variable-scattered-dropout-seed17-r10` | 16427 | finetune | 256 | 91.6188% |
| `bert-mini-fp16-ssd-b64-w256-aux-edges-scattered-dropout-seed17-r11` | 17677 | finetune | 256 | 91.5973% |
| `bert-mini-fp16-nvme-b64-w256-aux-edges-whole-dropout-seed17-r11` | 16177 | finetune | 256 | 91.5190% |
| `bert-mini-fp16-ssd-b64-w256-variable-seed17-r5` | 17927 | finetune | 256 | 91.4352% |
| `bert-mini-fp16-nvme-b64-w512-aux-edges-scattered-ftweak0.20-seed17-r12` | 13000 | pretrain | 512 | 91.0969% |
| `bert-mini-fp16-ssd-b64-w512-aux-edges-scattered-consistency0.10-seed17-r13` | 16677 | finetune | 512 | 91.0522% |

Selected `bert-mini-fp16-nvme-b64-w256-variable-whole-dropout-seed17-r10/best.pt`,
step 15,250. This is the older whole-modality-dropout model without the auxiliary
start/end heads. Its page-type F1 leads the 256 auxiliary-head scatter run by only
0.034 percentage points; this choice is specific to page-type mining.

## Paired context comparison

Both modes process exactly 431,396 unique catalogue rows from 1,979 complete
development documents, including all 3,758 human-labelled pages in `dev.jsonl`.
The original 193 manifest streams are concatenated in their existing recorded order
for continuous mode. Per-document mode resets inference at every catalogue boundary.
Long documents are still scanned with overlapping windows. Document identity is
never a model input. Both modes use window 256, stride 128, centre-weighted logit
merging, edge ramp 64, batch 32, CUDA FP16, and available pagination; augmentation
is disabled. The same classification metric implementation is used as in training.

| Mode | Curated weighted F1 | Accuracy | Macro F1 | Full windows / all windows | Inference seconds |
|---|---:|---:|---:|---:|---:|
| per_document | 90.4184% | 90.4470% | 70.9298% | 1904 / 3238 | 34.51 |
| continuous | 91.6585% | 91.6977% | 74.0692% | 3370 / 3370 | 29.10 |

Timing excludes checkpoint/data identity verification and panel construction;
it includes feature reads and inference. These are single measurements while another
training job was running, not a controlled throughput benchmark.

Continuous mode changes 9,422 / 431,396 page-type decisions. On curated pages it
changes 143: 80 become correct, 33 become wrong, and 30 change between wrong classes.
The accuracy gain is 47 / 3,758 = 1.251 percentage points.
Weak DB agreement F1 is 90.1342% per document versus 90.3278% continuously;
DB agreement is not a measure of verified correctness.

| Curated subset | Pages | Per-document F1 | Continuous F1 |
|---|---:|---:|---:|
| document_fits_window | 2700 | 89.7901% | 91.4116% |
| document_exceeds_window | 1058 | 92.0139% | 92.2894% |
| first_or_last_16 | 3288 | 90.8672% | 92.2633% |
| interior | 470 | 87.1671% | 87.1123% |

The observed gain is concentrated in shorter documents and document edges.
Use continuous inference for the initial dump-mining pass. This comparison uses
one fixed concatenation order; it does not establish invariance to neighbouring
documents. A different ordering is a separate future robustness check.

### Follow-up: is the gain at starts or endings?

Using disjoint edge regions (first/last `min(16, floor(length/2))` pages), the
curated-page errors break down as follows. The disjoint definition differs slightly
from the overlapping first/last-16 report above for very short documents.

| Region | Curated pages | Per-document errors | Continuous errors | Continuous fixes | Regressions |
|---|---:|---:|---:|---:|---:|
| Beginning | 1,293 | 111 | 111 | 16 | 16 |
| Ending | 1,990 | 188 | 141 | 61 | 14 |
| Interior | 475 | 60 | 60 | 3 | 3 |

The entire net accuracy gain of 47 pages is at document endings. Frequently
corrected errors include BackEndSheet predicted as BackCover (7 fixes), BackCover
as BackEndSheet (7), BackCover as NormalPage (6), and BackEndSheet as NormalPage (4).
This is not primarily a front-versus-back reversal problem.

Crucially, the selected r10 checkpoint predates explicit document-edge crops and
auxiliary start/end supervision: neither is enabled in its configuration. It has
variable random scan lengths, but these scans generally continue across document
boundaries or end at arbitrary positions within documents. An observed context
endpoint therefore does not reliably mean a true document ending during training.
Padding positions are masked out of attention; there is no explicit end-of-document
token. This makes reliance on following-document evidence plausible, but does not
prove it: inference also changes absolute window positions, available context length
and overlap merging. Before changing training, a useful next diagnostic is the same
paired comparison on r11, which already has explicit edge crops and auxiliary heads.

## Curated results per page type

F1 is a percentage; changes are percentage points. Tiny supports make several
rare-class movements unstable. Classes without curated support are omitted here
but remain in the machine-readable confusion matrices.

| Page type | Support | Per-document F1 | Continuous F1 | Change |
|---|---:|---:|---:|---:|---:|
| Advertisement | 17 | 68.57 | 80.00 | +11.43 |
| Appendix | 2 | 0.00 | 0.00 | +0.00 |
| BackCover | 276 | 89.52 | 93.38 | +3.86 |
| BackEndPaper | 6 | 66.67 | 61.54 | -5.13 |
| BackEndSheet | 276 | 91.91 | 96.70 | +4.79 |
| Bibliography | 5 | 57.14 | 57.14 | +0.00 |
| Blank | 154 | 87.66 | 87.30 | -0.37 |
| CalibrationTable | 495 | 99.70 | 99.80 | +0.10 |
| Colophon | 2 | 0.00 | 0.00 | +0.00 |
| Cover | 12 | 52.17 | 54.55 | +2.37 |
| Errata | 6 | 71.43 | 83.33 | +11.90 |
| FlyLeaf | 178 | 83.94 | 85.25 | +1.31 |
| FragmentsOfBookbinding | 1 | 0.00 | 0.00 | +0.00 |
| FrontCover | 266 | 88.72 | 89.91 | +1.18 |
| FrontEndPaper | 1 | 0.00 | 0.00 | +0.00 |
| FrontEndSheet | 44 | 85.39 | 84.44 | -0.95 |
| FrontJacket | 204 | 93.95 | 94.48 | +0.54 |
| Frontispiece | 11 | 84.21 | 84.21 | +0.00 |
| Illustration | 42 | 67.42 | 71.43 | +4.01 |
| Impressum | 2 | 66.67 | 40.00 | -26.67 |
| Imprimatur | 3 | 80.00 | 100.00 | +20.00 |
| Index | 136 | 97.42 | 97.06 | -0.36 |
| Jacket | 64 | 95.31 | 95.31 | +0.00 |
| ListOfIllustrations | 84 | 75.00 | 76.02 | +1.02 |
| ListOfMaps | 12 | 81.48 | 84.62 | +3.13 |
| ListOfTables | 4 | 54.55 | 54.55 | +0.00 |
| Map | 195 | 92.61 | 92.80 | +0.19 |
| NormalPage | 369 | 78.40 | 80.55 | +2.15 |
| Preface | 1 | 100.00 | 100.00 | +0.00 |
| SheetMusic | 133 | 99.25 | 99.25 | +0.00 |
| Spine | 193 | 99.22 | 99.48 | +0.26 |
| Table | 112 | 86.22 | 86.61 | +0.38 |
| TableOfContents | 183 | 94.77 | 94.12 | -0.65 |
| TitlePage | 269 | 93.24 | 94.52 | +1.28 |

## Reproduction and artifacts

Implementation: `page_sequence/compare_page_context.py`.
Command:

```bash
python -m page_sequence.compare_page_context \
  --checkpoint /home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-nvme-b64-w256-variable-whole-dropout-seed17-r10/best.pt \
  --manifest /home/ikohut/tmp/page-sequence-data/dev.jsonl \
  --training-store /home/ikohut/tmp/document-store-fp16-v1 \
  --output /home/ikohut/tmp/page-type-context-comparison-w256-r10-reproduction \
  --device cuda --batch-size 32
```

Original artifact directory:
`/home/ikohut/tmp/page-type-context-comparison-w256-r10/`.
Contains `report.json`, aligned FP32 class probabilities for both modes, and
`panel.npz` with canonical row IDs, document spans, labels and class order.
Predictions and the large panel are not committed to Git.

Checkpoint SHA256: `65c9e6b24d4cda906bfa9fc505199a41caa69d6b709c6c25ac2a2d756bd2a233`.
Manifest SHA256: `1a3779cb4c65c9b235122d49d69e576d39ead3d6ed5e5c768f3443e20b1f1b22`.
Catalogue SHA256: `4cc82720b3330744c92cc54102bb989502d0b9e5dfefeb8488f62a2f40afec51`.

Validation: 17 targeted tests passed, covering panel/order rejection, paired label
provenance, real overlap merging, full-window coverage, and existing evaluation
and short-context behavior.
