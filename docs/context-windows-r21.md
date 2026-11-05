# Context length experiments: fixed sampled data

The first three r21 runs train independent W1 models using S (180 grouped
statistics), T (new OCR-lines mmBERT), or I (DINO). They start from the same
original pretrained BERT-mini initialization as r20, then run both corpus
pretraining and fine-tuning. They do not start from a trained context checkpoint.
Pagination and five-bit scan/document metadata inputs remain unchanged.

## Sampling and optimizer updates

`sampling_window_size: 512` decouples the sampler from `window_size: 1`.
The sampler, synthetic scans, source quotas, stage policies, scan tiling and
64-parent-window batches are unchanged from r20. At each optimizer update:

1. Materialize and collate the same 64 sampled parent windows. Compute duplicate
   page/pair weights across the parent batch as before.
2. Apply pagination, whole-modality and scattered dropout to the parent batch.
   Whole-window dropout thus keeps its original scope; it is not independently
   redrawn for every singleton.
3. Split each parent into consecutive context chunks and keep every real page.
   Carry the original targets, duplicate weights and boundary feature vectors.
   Discard fully padded chunks. Do not invent scan/document edges at cuts.
4. Process all singleton chunks together, with a maximum physical batch of
   32,768 pages. Normalize each task's loss by its valid weighted target count
   across the entire parent batch.
5. Clip gradients and perform one optimizer/scaler update.

There are at most 32,768 page positions per optimizer update. Padding is excluded;
the actual number of real pages follows the parent sampler. Changing context size
does not increase the number of selected interesting or annotated anchors.
Coverage counters, budgets, early stopping, validation intervals and LR warmup
remain in parent optimizer-update units. Independently trained runs can still
stop at different stages/steps and have different dropout randomness.

The partitioning code also supports larger contexts such as 64/128/256. Pairwise
supervision at new chunk seams is excluded; no representation attends across cuts.
The current implementation rejects boundary-consistency loss for chunked runs.

The initial W1 S run used 1,024-page physical chunks. Following the isolated
benchmark on 2026-11-05, all three W1 recipes use a maximum physical batch of
32,768. S resumes from an audited checkpoint migration that changes only the
physical-batch configuration; the original checkpoint is preserved and all
other checkpoint fields are verified unchanged. T and I start with the larger
physical batch. Fresh CUDA preflight verifies the new setting before full training.

## W1 outputs and evaluation

The architecture and parameter count are retained. Page-type, side, auxiliary
start-page and end-page losses train normally. Pairwise boundary loss (`starts`)
is zero; its parameters receive no gradients or AdamW updates. Its predictions
and segmentation metrics are disabled, with zero evaluated pairs and null scores.

Full development scans are read in contiguous blocks for efficient I/O, but
every page is encoded independently at position zero. Only genuine scan
endpoints and supplied document metadata become input flags. Evaluation retains
the page-type/side per-class, front/back and document-region metrics, with both
unknown and supplied document metadata. W1 has no overlapping predictions to merge.
Context, missing-pagination, modality robustness and auxiliary-edge diagnostics
remain enabled. Final stream-versus-document comparison is omitted for this
singleton baseline.

An initial full checkpoint is saved before the first update, alongside the usual
phase-final, phase-init and selected checkpoints. Resume retains the parent
sampler, optimizer, scaler, RNG and ClearML task.

## Queue and artifacts

Recipes and frozen source: `/home/ikohut/tmp/context-windows-r21/`.

| Experiment | Slot | Scheduled after |
|---|---|---|
| r21-w1-S | NVMe | r20 SI |
| r21-w1-T | SSD | r20 TI |
| r21-w1-I | NVMe | r21 W1 S |

W512 modality experiments retain priority within their disk slots. W1 can use
the other disk's free slot while a W512 modality run finishes. The global limit
remains two simultaneous trainings; pending W512 work must not be displaced.

Jobs are registered in the existing
`page-sequence-modality-ablations-r20.service` queue. The global limit remains two
trainings, one per disk; each run retains eight loading workers. Training reads
the native store on its assigned disk, with no LMDB fallback.

Before full training, each job performs a six-update real-data CUDA smoke test
in its allocated slot. The test audits singleton inputs, permanently masked
modalities, finite losses/parameters, loss weights, unchanged boundary-head
weights, matching r20 initial model parameters, 512-page sampler state, disabled
pairwise evaluation and full phase snapshots. Failed preflight blocks the job
for diagnosis. Smoke artifacts are separate from full-run metrics and ClearML.

`tools/prepare_context_window_experiments.py` creates the frozen recipes and
`jobs.json`; registration in an existing monitor is a separate operation. Running
r20 sources are never edited. Results for all registered jobs are summarized
when the queue completes; original r20 and r21 IDs remain distinguishable.
