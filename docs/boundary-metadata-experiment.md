# Optional document-boundary metadata experiment r16

This fresh 256/SSD run repeats the r15 scan-only recipe with one training change:
complete archive document-boundary metadata is supplied for 50% of constructed
scans and withheld for the remaining 50%. It starts from pretrained BERT Mini,
not a resumed r15 optimizer. Seed 17, batch 64, eight loaders, FP16, scan sampling,
pagination/modality/scattered dropout, auxiliary heads, losses and training budgets
otherwise match [r15](boundary-feature-experiments.md).

The unchanged five-bit input is
`[scan_start, scan_end, metadata_provided, document_start, document_end]`.
Scan flags are always retained. Metadata availability is chosen once per whole
scan by the checkpointed sampler RNG, before tiling; all overlapping tiles share
the choice. Supplied scans carry catalogue-derived positive and negative start/end
values on every page, simulating a complete archive-provided annotation. Absent
scans have all last three bits zero. Targets remain supervised in both modes.
Only metadata availability is augmented; feature dropout preserves all five bits.

The goal is unchanged or better performance without metadata, plus an additional
benefit when supplied. This is an experimental goal, not a guaranteed property.
Checkpoint selection and early stopping continue to use ordinary absent-metadata
curated development page-type weighted F1. Every scheduled validation additionally
runs the same development streams with complete metadata supplied; full reports
are nested under `metadata_provided` in validation JSON, with ClearML summaries
under `dev_metadata_provided`. No test data are used. At completion, per-document
versus continuous comparisons run separately with metadata absent and supplied.

The supplied-mode boundary score measures using provided information, not inferred
boundary quality. Its absent-mode counterpart remains the relevant boundary
inference measure. Input metadata is not automatically an output override.

Runtime: `/home/ikohut/tmp/boundary-metadata-r16/w256/`.
Run: `/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-ssd-b64-w256-boundary5-metadata50-scattered-seed17-r16/`.
Service: `page-sequence-w256-boundary-metadata-r16.service`.
Data reads use the published native FP16 SSD store; old LMDB paths remain provenance.
The frozen source copy and checksums are saved beside the runtime configuration.
The existing supervisor retains its 2-GiB available-memory stop guard and does not
retry failed runs automatically.

Validation before launch: 197 regression tests passed. The frozen source at
`298d050` passed four real-data CUDA FP16 optimizer updates with batch 64 and
8 loaders; all parameters and gradients were finite and the scaler stayed at
1024. The smoke run consumed 130 supplied-metadata and 126 absent-metadata windows
from 66 supplied and 58 absent constructed scans, and completed two dual-mode
validations. Details are saved in runtime `smoke-verification.json`.

Full training was verified advancing at step 14 with finite losses and
gradients, loss scale 1024, both metadata modes sampled, and zero
fallback storage reads. [ClearML task](http://semant.cz:8090/projects/cf7452f7555d4335a213e1f4159b80a9/tasks/b9f269d8e1804840b350b75bbca793db/output/log).

## 512-page NVMe companion

Launched on 2026-10-28 alongside the ongoing r15 512 scan-only and r16 256
metadata experiments. The new run uses the identical frozen source (`298d050`)
and metadata policy as the 256 run, with window 512, stride 256 and edge ramp
128. Batch size 64, eight loaders, training budgets and other settings remain
unchanged. Initialization is fresh pretrained BERT Mini, not a resumed optimizer.
The native store is `/home/ikohut/tmp/document-store-fp16-v1`; pagination is read
from `/home/ikohut/tmp/pagination-experiment-w128/pagination.npy`. No LMDB fallback
is enabled. Both metadata evaluation modes and final context comparisons remain on.

Runtime: `/home/ikohut/tmp/boundary-metadata-r16/w512/`.
Run: `/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-nvme-b64-w512-boundary5-metadata50-scattered-seed17-r16/`.
Service: `page-sequence-w512-boundary-metadata-r16.service`.
The new job's supervisor stops it when available RAM stays below 3 GiB for ten
seconds, ahead of the older jobs' 2-GiB guards. It does not retry automatically.

Its real-data smoke test passed four CUDA FP16 optimizer updates at batch 64 with
eight loaders and two dual-mode validations. All parameters and gradients remained
finite; loss scale stayed at 1024; fallback reads were zero. It sampled 170 supplied
and 86 absent metadata windows. This small sample is not an estimate of the long-run
50% per-scan probability. Around 7.4 GiB RAM remained available during the check
with the two existing jobs active. Source checksums and smoke verification are
saved with the runtime configuration.
