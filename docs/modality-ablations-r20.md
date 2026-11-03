# Six modality ablations of the 180-stats W512 recipe

Reference: [r19](stats-fine-groups-r19.md), new OCR-lines mmBERT and 180-dimensional
statistics, using optional complete document metadata. Each ablation starts fresh
from the same pretrained BERT Mini and seed 17; it does not fine-tune the trained
r19 checkpoint.

| Variant | Retained main signals | Storage slot |
|---|---|---|
| S | Statistics | NVMe |
| T | Text | SSD |
| I | Image | NVMe |
| ST | Statistics + text | SSD |
| SI | Statistics + image | NVMe |
| TI | Text + image | SSD |

Pagination and the five-bit scan/document-boundary input remain in every variant.
Their availability/dropout policy is unchanged. Thus “S only” means only statistics
among S/T/I, not statistics without those auxiliary inputs.

## Controlled change

`enabled_modalities` in the run config permanently excludes the other main
signals. A dataset wrapper zeros their vectors and availability flags **before**
augmentation, so whole-modality dropout's rescue cannot restore excluded signals.
Naturally missing retained signals stay missing; scattered dropout may still
remove all retained signals on a page. The original whole-modality, pagination
and scattered-dropout probabilities remain unchanged. With fewer available
modalities, the original rescue rule naturally changes the effective subset
distribution; no new dropout policy is introduced.

The model specification also retains the subset and enforces its availability
mask during forward inference, even if an external caller supplies all vectors.
Biased/nonlinear branch outputs remain gated by presence. The complete architecture,
dimensions and parameter count stay fixed; this is an information ablation.
All six smoke models' initial parameter hashes are checked for equality.

Everything else follows r19: context 512, batch 64, eight loaders, FP16 compute
with FP32 parameters, 50% complete metadata scans, 180 stats with `1/sqrt(180)`
scaling, the same sampler, labels, split and development panel, auxiliary start/end
heads, loss weights, learning rates, 15,427/10,000 update budgets and early stopping.
Weak page-type weight is 0.2 during pretraining and 0.05 during fine-tuning.
The original frozen r19 validation inference/merging implementation is retained.
Phase-end and next-phase initialization checkpoints remain protected.

Data are read from the new native store on each assigned disk. Kolosus LMDB paths
remain provenance, with no training fallback. The native reader still reads the
same physical records; this study does not claim an I/O throughput improvement.

## Runtime and queue

Root: `/home/ikohut/tmp/modality-ablations-r20/`.
Each variant has its own directory, configuration, smoke run and ClearML task.
All share the frozen `code/` snapshot. Config differences, code hashes, baseline
identity and the runner patch are retained in the root/variant directories.

Queue order by slot:

- NVMe: S → I → SI.
- SSD: T → ST → TI.

`page-sequence-modality-ablations-r20.service` manages all six with a **global
maximum of two trainings and at most one managed experiment per storage slot**.
External `page_sequence.runner train` processes also count against the global cap.
Launches are spaced by 240 seconds to let memory use stabilize. New launches need
9 GiB available RAM; the existing low-memory guard interrupts the newest managed
attempt if available memory stays below 3 GiB. Consequently a second slot may
wait for RAM even when the GPU has room.

Transient worker/OOM interruptions resume from checkpoints, appending to each
run's existing ClearML task. Repeated or deterministic failures remain visible
as `needs_diagnosis`; other eligible experiments can continue. Slot checks apply
to retries as well as first launches. No incomplete job is labelled finished.
The service survives logout and is enabled after preflights pass. It exits when
all six training units have completed their final reports successfully.

Live queue state: `state.json`; history: `events.jsonl`; monitor log: `monitor.log`.
Per-variant full logs: `<variant>/managed-training.log`; failures:
`<variant>/last-failure.json`. Actual training artifacts live under
`/home/ikohut/tmp/page-sequence-runs/`, with names containing
`stats180-ablS`, `stats180-ablT`, etc., and suffix `seed17-r20`.

After completion, `tools/summarize_modality_ablations.py` automatically produces
`results.json` and `results.csv`, selecting each run by fine-tuning development
curated page-type weighted F1, as before. The JSON retains all per-class and
boundary metrics plus supplied-metadata scores and ClearML links. The summary
requires every queue job and retained phase checkpoint to be complete.

## Validation before full training

316 regression tests passed, including all six subsets, dataset worker
serialization, dropout-rescue isolation, model-inference invariance to excluded
content/presence, zero gradients for excluded branches, checkpoint-specification
round-trip and one-experiment-per-storage-slot scheduling.

Each variant receives a separate real-data CUDA smoke test with four pretraining
and two fine-tuning updates, batch 64 and eight loaders. Forward audits cover both
training and validation and assert excluded vectors/presence stay zero. Verification
checks finite losses, gradients and parameters, correct stage weights, zero LMDB
fallback, both metadata modes and protected phase snapshots. ClearML is disabled
for smoke. `smoke-verification.json` records all six audits and identical starting
parameter hashes; only after it exists does the handoff enable the full queue.

An initial smoke-helper attempt was stopped because its multiprocessing entry
point lacked a main guard. Its artifacts are retained in S's
`smoke-run-entrypoint-attempt/` and `smoke-entrypoint-attempt.log`; the guarded
helper was restarted cleanly. No full experiment used that attempt.

The presence of configuration or an active preflight does not mean full training
has started. Use `state.json` and each run's metrics/managed logs for actual progress.
