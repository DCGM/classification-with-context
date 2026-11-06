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

The launcher configures CUDA expandable segments and a 0.3 cache collection
threshold. Variable real-page counts otherwise let the default CUDA allocator
retain far more memory than a single full batch needs, preventing the second
W1 process from fitting. This is a runtime allocation policy; sampling, weights,
optimizer and checkpoint configuration are unchanged. Smoke reports record the
allocator policy and peak allocated/reserved GPU memory.

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

On 2026-11-05 the four mixed combinations were added so that W1 covers all seven signal
subsets, like the W512 ablations. `tools/prepare_context_window_experiments.py
--add-combinations` creates ST, SI and TI from the r20 recipes of the same name, and STI from
r20 SI with all three signals enabled explicitly: r20 has no STI recipe, and its STI reference,
r19, is the same config without `enabled_modalities`. Each recipe reads the native store on its
own disk, shares the frozen r21 code, and gets the same W1 changes, smoke test and expected
initial parameters as S/T/I. They are listed in `jobs-combinations.json`. W1 results join the
[modality overview](modality-overview.md) once all seven runs are finished.

Recipes and frozen source: `/home/ikohut/tmp/context-windows-r21/`.

| Experiment | Slot | Scheduled after |
|---|---|---|
| r21-w1-S | NVMe | r20 SI |
| r21-w1-T | ~~SSD~~ NVMe | r20 TI |
| r21-w1-I | ~~NVMe~~ SSD | ~~r21 W1 S~~ r20 TI |
| r21-w1-ST | NVMe | r21 W1 T |
| r21-w1-SI | SSD | r21 W1 I |
| r21-w1-TI | NVMe | r21 W1 ST |
| r21-w1-STI | SSD | r21 W1 SI |

On 2026-11-05, r21-w1-T was paused at its step-6,000 checkpoint (a copy is retained in
`T/pause-20261105/`) so that r21-w1-I could take the SSD slot. Before I started, its
recipe was relocated to the SSD copy of the native store: `index`,
`training_store.path`, `run_dir` and the ClearML name changed only, with the
backup and before/after record in `I/storage-relocation-*`. Both store copies
have identical manifests, checksums, plans and shard listings, and the relocated I
recipe now differs from T only in its modality and names. T stays in the queue with
phase `paused` and resumes from its checkpoint once it is set back to `retry_wait`.

After r21-w1-S finished (19:07), T resumed from step 6,000 in the free NVMe slot, in its existing run
directory and ClearML task. `tools/relocate_training_store.py` moved it to the NVMe copy of the store:
it checks that both copies have byte-identical metadata files and the same shard names and sizes, then
changes only the config's `index` and `training_store.path` (also in the smoke config) and the
checkpoint's recorded store path, verifying every other checkpoint field unchanged. The originals are in
`T/store-relocation-20261105T195420/`; the runner rehashes the catalogue at resume. The run directory and
ClearML name still say `ssd`.

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

## W1 I against r19 STI W512 (2026-11-06)

For the 2026 progress report, r21-w1-I's `finetune_best.pt` was evaluated like the
[modality overview](modality-overview.md) (`tools/modality_overview.py evaluate`, condition `none`: no page
numbers and no document boundaries, 193 full dev streams), into `/home/ikohut/tmp/pz2026-page-type-overview/`,
kept out of the overview directory. Against r19 STI W512 in the same condition (values in %):

| Model | Type P / R / F1 | Side P / R / F1 | Border P / R / F1 | Curated front→back / back→front errors |
|---|---|---|---|---|
| r21 I W1 | 79.6 / 79.8 / 79.5 | 76.7 / 76.7 / 76.7 | – | 31 / 795 (3.9), 32 / 558 (5.7) |
| r19 STI W512 | 91.5 / 91.6 / 91.4 | 88.8 / 88.8 / 88.8 | 95.4 / 86.4 / 90.7 | 2 / 795 (0.3), 8 / 558 (1.4) |

The largest per-type F1 gains (at least 40 curated pages) are FrontEndSheet (19.0 → 86.4), Blank, BackEndSheet,
ListOfIllustrations and FrontCover; CalibrationTable, SheetMusic, Spine, Jacket, Map and Index are already above
90 with W1 and change by less than two points. The figure and table of the report are generated by
`~/Projects/Metakat/PZ_2026_MetaKat_obrazky/vysledky_typy.py`.
