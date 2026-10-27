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
