# Page-stream experiments

PyTorch owns training, batches and checkpoints. Hugging Face Transformers supplies BERT's
implementation/configuration/pretrained weights. There is no tokenizer: **one page vector is one
token**. The initial checkpoint is `google/bert_uncased_L-4_H-256_A-4`, pinned to
`387825ce42dbb39b87911cdf8e383ee3b25184f8`; Transformers is pinned to 4.57.6.
The [experiment plan](../docs/sequence-experiments.md) explains the hypotheses and later comparisons.

Install into the project environment:

```bash
python -m pip install -e '.[experiments,sequence,db,dev]'
```

## Module boundaries

| Module | Responsibility |
|---|---|
| `schema.py` | Page/document/window contracts and label order |
| `index.py` | Streaming CSV ingestion, disk-backed page index, order/split/copy audits |
| `sampling.py` | Fit-only coverage, interesting and annotated windows; stage quotas and resumable state |
| `features.py` | Lazy read-only vector LMDBs, per-modality normalization and saved feature schema |
| `data.py` | Dynamic/fixed window datasets, targets, independent masks and padding |
| `models.py` | Fusion, backbone adapters and three shared output heads |
| `losses.py` | Independently normalized masked losses and duplicate-occurrence weighting |
| `inference.py` | Pair-covering windows, overlap merging, arbitrary scan tagging and segment IDs |
| `manifests.py` | Frozen canonical evaluation streams and validation |
| `evaluation.py` | Metrics after merging, annotation/DB separation, start/span/stream metrics |
| `context_evaluation.py` | Fixed development panels, batched short scans and bounded diagnostic time |
| `runner.py` | PyTorch training, development selection, checkpoint/resume, evaluation/prediction CLIs |
| `smoke.py` | Synthetic metadata/vector-store fixture and end-to-end run |

To add another sequence architecture, implement a backbone with `hidden_size`, `initial_scale`,
`forward(page_embeddings, attention_mask) -> (B,L,H)` and `specification()`; extend the small
construction/restore factory. Fusion and heads can remain shared. A hierarchical encoder may do
its internal pooling/decoding and return page-aligned states through this contract. A model with
different heads can instead return the same output dictionary and reuse data/losses/inference.
No backbone-specific logic belongs in the sampler, dataset, merger or evaluator.

## First BERT Mini configuration

[configs/bert_mini.json](configs/bert_mini.json) is the pilot template. Replace vector-store paths,
index/manifest/run paths, and benchmark the proposed batch/update budgets before a corpus run.
The 1,000 + 250 update budgets are a pilot, not a claim of whole-dump coverage.

Dense inputs: 768-dimensional DINOv2 and mmBERT, independently L2 normalized in float32.
Statistics: all **147 grouped features**, bounded by the existing fixed normalizer, divided by
sqrt(147). Raw sparse LMDB vectors currently have 3,324 coordinates; this is a distinct dimension.
Three explicit modality-presence flags join the inputs. Source availability such as `has_ocr_xml`
is retained within the statistics vector. Missing vectors are zero with a false presence flag;
malformed/nonfinite vectors fail instead of being silently treated as missing.
OCR coordinate fractions extending beyond the page are clipped to [0,1] during normalization;
negative counts and other invalid measurements still fail. Stored vectors are unchanged.

Separate modality MLPs -> fusion LayerNorm -> learned scale -> BERT Mini (first corpus experiment). BERT adds its own positions
and constant segment-0 embeddings; known document IDs never enter the network. The unused word
table is replaced by a frozen one-row placeholder; the pooler is omitted. With these feature
dimensions the separate-MLP tagger has **4,196,140 parameters** (joint-linear: 3,997,228). The scale starts from the checkpoint's
centered word-vector RMS (approximately 0.05093 for this revision).

Config changes for comparisons:

- P0: `"backbone": {"kind": "page", "hidden_size": 256}`.
- P1: pretrained BERT Mini with joint linear fusion (`"fusion": "linear"`).
- P2: `"pretrained": false`, retaining checkpoint/revision for the identical architecture.
- P3 (first corpus run): `"fusion": "separate_mlp"`. `"joint_mlp"` is also available for the subsequent capacity comparison.

Outputs are `type_logits (B,L,39)`, `placement_logits (B,L,3)`,
`starts_document_logits (B,L)`, plus a boundary `pair_mask`. Start output i describes page i.
Position zero in each model window is invalid for the start head. Padding is separate from every
task's supervision mask. Curated type overrides DB type throughout; missing placement is ignored.
Loss defaults are gold type + 0.2 weak type + 0.5 placement + 0.5 starts, each with its own valid
weighted denominator. Duplicate row targets in the optimizer batch receive reciprocal occurrence
weights; start targets deduplicate by predecessor/current row pair because different synthetic
predecessors are different examples.

## Prepare metadata before reading vectors

Create a JSONL document manifest, one entry per retained leaf document:

```json
{"library":"library-id","doc":"issue-a","split":"fit","volume":"volume-1","periodical":"periodical-1","copy_group":"copy-family-a"}
{"library":"library-id","doc":"issue-b","split":"fit","volume":"volume-1","periodical":"periodical-1","copy_group":"copy-family-b"}
```

Splits are `fit`, `dev`, `test`. Preserve the existing final test documents; assign whole
document/copy groups to development before building the index. This module consumes explicit
split assignments; automatic copy discovery and a class-stratified split planner are future
preparation tools. Unknown volume/periodical values can be omitted. Supply verified hierarchy
IDs: a generic parent/collection ID is not sufficient evidence that two documents are issues in
one volume. `copy_group` can be omitted for documents without known copies.

Build the SQLite index using the repository's headerless 13-column CSVs:

```bash
python -m page_sequence.index \
  --documents /home/ikohut/tmp/page-sequence-data/documents.jsonl \
  --context /path/to/train-context.csv /path/to/test-context.csv \
  --annotations /path/to/train-annotations.csv /path/to/test-annotations.csv \
  --output /home/ikohut/tmp/page-sequence-data/index.sqlite
```

CSV input streams into disk-backed staging tables. The final page index is physically ordered by
document and numeric page order. The builder rejects order ties, duplicate document/page rows,
unknown curated labels, annotations absent from context, pages missing split assignments, and
page UUIDs/known copy groups crossing splits. Missing or unsupported DB types simply have no
type target. Failed builds retain a `.building` file for diagnosis and never publish a usable
index; use a fresh output path after correcting metadata. Budget disk space for staging/sorting;
SQLite temporary storage should also be on a permitted writable disk (`SQLITE_TMPDIR` may be set
under `/home/ikohut/tmp/` before launching).

Runtime page lookups fetch up to 512 unique rows per query and restore requested order and
duplicate occurrences afterward. The sampler reuses document IDs from those same records for
boundary accounting and fetches anchor/document identities together. Neighbor pools are built
only when requested, preserving candidate order and RNG draws. These changes do not precompute
training windows or keep the page catalogue in RAM; only the existing document catalogue is held.

The index builder screens documents with a deliberately simple internal gappiness estimate:
`missing / span`, where `span = last_order - first_order + 1` and
`missing = span - observed_page_count`. Documents above **30%** are excluded by default;
override with `--max-gap-fraction`. For example, `0,2,4,6` has 3/7 missing positions and is
excluded, while `0,1,3,4` has 1/5 and is retained. Leading/trailing unknown pages are not
estimated; a singleton has zero measurable internal gaps. No attempt is made to repair indexing.

All measurements and original splits remain in the SQLite `document_gap_audit` table.
The CLI also exports exclusions to `OUTPUT.ignored.jsonl` (override with `--ignore-list`) and
reports excluded document counts per split. The same rule applies to fit/dev/test: excluded
test documents remain identified as test in the audit and must never be reassigned to training.
Report this evaluation exclusion when comparing against the original test population.
This screen does not yet insert placeholder slots for the smaller gaps in retained documents.

```bash
python -m page_sequence.manifests \
  --index /home/ikohut/tmp/page-sequence-data/index.sqlite --split dev \
  --target-pages 2048 --recipe siblings --seed 17 \
  --output /home/ikohut/tmp/page-sequence-data/dev.jsonl
```

This freezes scan order, retaining complete internal documents and preferring sibling runs.
`--recipe shuffled` gives an alternate ordering. Individual long documents remain whole, so
requested length is a target, not a cap. Prepare separate 512/2048/8192-page stress manifests;
evaluate each separately instead of pooling their repeated curated pages. The manifest validator
checks split membership, canonical row uniqueness, contiguous document order, full internal joins
and joins between known copies. Outer crop censoring is derived from indexed endpoints.

Training window sources use 60/25/15 coverage/interesting/annotated quotas, changing to 20/15/65
at fine-tuning. Interesting draws split approximately 2/3 rare-type anchors and 1/3 starts.
Anchor class weights are capped sqrt inverse frequency. All visible annotations are used.
Targeted neighbor proposals request 60% same-volume, 20% cross-volume same-periodical and 20%
unrestricted joins, with recorded fallback when candidates cannot be found. Coverage traverses
every fit document once per pass in shuffled volume runs; its join mix is measured rather than
forced to match the targeted proposal ratios. Targeted repeats never advance coverage. Logs
include actual boundary-window exposure, realized strata, unique join volumes/pairs, class/gold
exposure and completed coverage passes. Inspect these before adjusting source/neighbor quotas.

## Train, resume, evaluate

```bash
python -m page_sequence.runner train --config /path/to/edited-bert-mini.json
python -m page_sequence.runner train --config /path/to/edited-bert-mini.json \
  --resume /home/ikohut/tmp/page-sequence-runs/bert-mini-pilot/latest.pt
```

`--stop-after N` stops at absolute update N and saves resumable state. The first corpus
configuration uses FP32, 64 windows of up to 128 pages, and eight loading workers with one
prefetched batch each. Sampling stays in the parent process. Loading segments end at validation,
stage changes and the requested stop point; every prefetched draw is consumed before checkpoints.
This preserves exact sampler resume without retaining per-batch copies of the document catalogue.
A separate DataLoader generator prevents worker startup from changing model/dropout RNG state.

Training uses AdamW, differential learning rates, warmup and gradient clipping. Optional
`stopping.pretrain` / `stopping.finetune` settings accept `patience` (validation events) and
`min_coverage_epochs`. `pretrain_validate_every` / `finetune_validate_every` override the common
interval. With a pretraining coverage floor, only eligible validations compete for
`pretrain_best.pt`. Fine-tuning starts from that selected checkpoint's weights and optimizer
state, retaining the current sampler coverage/RNG and changing sampling quotas. Stage/update
budgets remain explicit upper limits. `latest.pt` saves stage progress and plateau counters.

ClearML is optional (`clearml.enabled`); configuration, provenance, scalar metrics, console and
resource usage are logged to the configured API. Automatic framework/model uploads are disabled,
`output_uri=False`, and checkpoints stay in `run_dir`: no ClearML file server is needed.
`clearml.json` records the task link. Exceptions mark the task failed. Resume reuses the task.

`python -m page_sequence.prepare` prepares the full corpus from the local metadata CSVs and the
existing gap audit, with a deterministic approximately 15% development split by document UUID.
Identical document UUIDs across libraries stay together, and the strict builder additionally
rejects any page UUID crossing splits. Documents with missing order, excessive gaps, or overfull
index counts are quarantined. Verified issue/volume relationships are fetched in metadata-only
batches before training. `dev.full.jsonl` covers the full development split; `dev.jsonl` monitors
all annotated development documents plus up to two held-out siblings per represented volume.
`test.jsonl` stays untouched during selection. Smaller retained order gaps currently use observed
pages without inserting missing-page placeholders; they do not create positive boundary labels.
The builder uses disk-backed staging and reclaims staging space when publishing the index.

Every validation uses merged full development streams. `best.pt` is selected by curated type
weighted F1; `latest.pt` is saved before evaluation and again after successful validation. A failed evaluation
leaves a checkpoint marked with pending validation; resume retries that validation before drawing
more training windows. Saved state includes optimizer,
sampler coverage/RNG/counters, Python/NumPy/torch RNGs, config, labels, feature schema,
checkpoint configuration, code hash, package versions, index/manifest hashes and cheap vector
file signatures (sizes/mtimes and sidecar metadata hashes). Exact resume requires these to agree.
These signatures assume vector stores stay immutable during a run; they are not full LMDB hashes.
Load only trusted local `.pt` files, which contain pickled training state.

```bash
python -m page_sequence.runner evaluate \
  --checkpoint /path/to/best.pt --manifest /path/to/dev.jsonl --split dev --device cuda \
  --output /home/ikohut/tmp/page-sequence-runs/dev-metrics.json \
  --predictions-dir /home/ikohut/tmp/page-sequence-runs/dev-predictions
```

An optional `--inference-config settings.json` compares merge settings on development streams.
Defaults: window 128, stride 64, center-weighted logit mean, 32-page edge ramp. Other merge values
are `uniform_logits`, `uniform_probs`, `center_probs`, `central`. Boundary weights use the minimum
of the two page weights and exclude window position zero. Real scan endpoints incur no artificial
context penalty. Stride must be smaller than the window to cover every adjacent pair, including
seams. Probabilities and segment IDs are decoded once after merging; first start probability is
NaN with a false validity mask, and segment IDs begin at zero by convention.

Metrics count each manifest occurrence after merging. Curated page types, weak DB agreement and
placement DB agreement are separate. Reports contain classification confusion/per-class scores,
overall start precision/recall/F1/AP, false splits, positive-join recall by hierarchy stratum,
exact uncensored spans, exact stream fraction, per-stream length and unique-page throughput.
ClearML logs accuracy, weighted precision/recall/F1, macro F1, and precision/recall/F1/support
for every page type and each of left/right/single, plus labelled confusion matrices (rows=true,
columns=predicted). Curated labels and weak database agreement remain separate. Exact span and
stream segmentation metrics and within-document false splits are also logged.

`front_back_confusion` counts front-as-back and back-as-front type errors separately, with
support and rates, for curated and weak labels independently. Front types are FrontCover,
FrontEndPaper, FrontEndSheet, FrontJacket, Frontispiece and TitlePage; back types are BackCover,
BackEndPaper and BackEndSheet. The combined rate divides reversal errors by all labelled
front/back pages, including correct predictions and other errors. Ambiguous Cover, Jacket,
FlyLeaf and movable content are excluded.

`front_back_position` flags front predictions in the last quarter and back predictions in the
first quarter of each **true** document, using observed page rank. Only uncensored documents
with at least four pages count; each rate divides flagged predictions by all predictions of
that direction in eligible documents. These are structural warning rates, not confirmed label
errors: unusual layouts can be legitimate. They do not use predicted boundaries, so boundary
mistakes cannot move the reference positions. Predictions on unlabelled pages can count here.

`side_sequence_db_agreement` distinguishes sustained left/right phase reversal from isolated
mistakes. It reports the fraction of all left↔right swap errors belonging to reversed runs of
at least four pages, their count, and the longest reversed run. Runs require true sides to
alternate and predictions to be the opposite on every page. True document changes, page-order
gaps, missing/single reference sides and non-alternating reference transitions break runs;
a correct or predicted-single page also breaks a reversed run.

For maximal reference-alternating blocks of at least four consecutive pages it also reports
raw `block_accuracy` and `oracle_phase_corrected_accuracy`: the latter chooses, independently
per block using the reference labels, whether to swap all predicted left/right labels.
Predicted single remains an error. A large gain indicates that much of the problem is phase
selection rather than failure to follow alternating progression. `reversed_block_fraction`
counts blocks where swapping improves agreement; `perfectly_reversed_blocks` counts fully
opposite blocks. Counts and denominators accompany these rates, and per-stream diagnostics
remain in the JSON report. These are diagnostics against database sides, not an inference
correction or a claim about gold labels. Censored document fragments are eligible, so they
do not establish that a reversal started at the actual document beginning.

A positive-only join stratum has no negative pairs from which to estimate precision: compare
complete fixed hard-join suites for precision/false-split behavior as well as stratified recall.
Bootstrap intervals, seam-distance/tiling-offset diagnostics, and automatic threshold calibration
remain later evaluation extensions. The template's 0.5 threshold is provisional; choose the final
operating point on development before test evaluation.

## Tag a scan without document identities

The `tag_pages(model, features, pages, config, device)` API accepts caller-ordered Page metadata
with empty `doc` values. No document index or true boundaries are required. CLI input is JSONL:

```json
{"library":"library-id","page":"page-uuid","path":"public.256/document/page.jpg"}
```

```bash
python -m page_sequence.runner predict --checkpoint /path/to/best.pt \
  --pages /path/to/scan.jsonl --device cuda --output /home/ikohut/tmp/scan-tags.npz
```

`--sources stores.json` can point to new embedding LMDBs using the same dimensions and frozen
normalization. Output retains input order and repeated page occurrences, page keys, all three
probability arrays, class IDs/label vocabularies, boundary validity and segment IDs.

## Verify before SSD arrival

Benchmark an existing real-data index and configured vector stores with:

```bash
HF_HOME=/home/ikohut/tmp/huggingface python -m page_sequence.benchmark \
  --config /path/to/config.json --batches 64 --gpu \
  --output /home/ikohut/tmp/reader-benchmark.json
```

This measures first/repeat loading, cached component decoding, GPU training on an already-loaded
batch, and synchronous training on different windows where the sample permits. It uses the
current reader without prefetch, materializes one batch at a time, and reports observed feature
availability, disk-read bytes, process memory and page faults. No OS caches are flushed. Random
128-page windows within documents measure loading mechanics, not the complete training sampler;
the full-corpus document catalogue memory and sustained epoch throughput need separate checks.
Use finished, immutable stores for the final preflight; concurrent extraction affects results.

Compare synchronous training against eight loader workers on the same warmed windows:

```bash
HF_HOME=/home/ikohut/tmp/huggingface python -m page_sequence.benchmark_workers \
  --config /path/to/config.json --workers 8 --batches 128 \
  --output /home/ikohut/tmp/worker-benchmark.json
```

This uses one CPU thread and one prefetched batch per worker, pinned memory, and the spawn
start method. Startup/warmup is reported separately; combined proportional worker/parent memory
is sampled outside the timed loop. The production runner uses the same worker count/prefetch settings, with finite segments to
preserve its checkpoint/resume contract.

For a broad index with curated training pages and verified issue/volume relationships, measure
the actual online sampler without warming the vector working set:

```bash
HF_HOME=/home/ikohut/tmp/huggingface python -m page_sequence.benchmark_sampling \
  --config /path/to/config.json --workers 8 --batches 512 \
  --output /home/ikohut/tmp/sampling-benchmark.json
```

It runs pretraining then fine-tuning mixtures, including targeted document joins. Sampling stays
in the parent process and is included in end-to-end timing. Reports separate sampler CPU time,
feature availability, measured/unique pages, realized join/source exposures, aggregate process
disk reads and physical reads on the image-vector store's block device. Caches are not flushed;
the second stage naturally benefits from the first. No checkpoint is saved.

```bash
python -m page_sequence.smoke --output-dir /home/ikohut/tmp/page-sequence-smoke --steps 80
python -m pytest -q --basetemp=/home/ikohut/tmp/page-sequence-tests
```

The smoke command creates synthetic vector LMDBs and metadata, then trains a small random BERT
through both sampler stages without downloading weights. Tests cover split/order rejection,
coverage and exact sampler resume, independent loss masks, missing vectors, padding/batch
isolation, overlap/seam coverage, prediction alignment, document-free inference, full-stream
scoring, and bit-identical CPU optimizer/checkpoint resume. Synthetic success checks mechanics;
no corpus quality, optimal context length or whole-dump training run has been measured. Storage
throughput has been benchmarked separately; see the experiment artifacts outside the repository.

Streams without any valid targets for a classification task contribute zero observations to that
task; they still contribute to the other tasks and boundary metrics. Such empty task masks are
expected with partially labeled development data and are skipped before calling sklearn metrics.

### Export to MetaKat

Export a trusted local checkpoint as an inference artifact, without training
state or data-store dependencies:

```bash
python -m page_sequence.export --checkpoint /path/to/best.pt --output /path/to/context.pt
```

The artifact contains weights, model specification, feature normalization,
class order and inference settings. MetaKat's `page_type_core_engine_context`
loads it with `weights_only=True`. Supply the matching LightlyTrain DINO export
and a local copy of the training mmBERT revision separately. Its binder writes
`pageType`, canonical `side`, and optional `startsDocument` for every scan page.
# NVMe and FP16 runs

`training_store` optionally selects the document-contiguous reader; see
[`training_store/README.md`](../training_store/README.md). Use its catalogue as
`index`. CUDA `precision: "float16"` enables FP16 autocast with FP32 master
parameters/AdamW and losses, checkpointed dynamic loss scaling, and FP32
overlapping-window accumulation. Default `float32` behavior remains available.

Sampling uses bounded document-page UUID sets for exact shared-page checks,
instead of a corpus-index join for every candidate pair. It also caches the
eligible cross-volume lists in their original order. Each cache retains at most
128 entries and 65,536 page IDs/document references; oversized documents retain
the SQL path, scanning the shorter side. These derived caches are not checkpoint
state, consume no RNG draws, and do not change the sampled windows or exclusions.


## Variable scan lengths and bounded context diagnostics

Set `variable_lengths: true` to choose a target length before sampling a window:
60% maximum length, 20% uniformly from 32 to maximum minus one, 10% from 8–31,
5% from 2–7, and 5% single pages. Unsupported ranges are removed and probabilities
renormalized for small synthetic fixtures. Coverage cursors advance by the actual
length; anchored samples retain their selected page. A boundary anchor has a
predecessor whenever the requested length exceeds one. Singletons have type and
side supervision but no boundary loss. The policy is used in both stages, saved
in sampler state, and checked on resume. Length counters record requested and
actual sizes. Short examples are right padded with masked attention and losses.

Optional `context_evaluation` settings are `enabled`, `sample_pages` (2048),
`budget_fraction` (0.1), and `final_sample_pages` (8192). The periodic diagnostic
partitions the same fixed development blocks into independent scans of lengths
1, 2, 4, 8, 16, 32, 64, 128 and optionally 256, up to the training maximum.
Blocks rotate curated, boundary and uniform anchor selection across dev streams;
this small enriched panel is diagnostic, not a representative corpus estimate.
Type, side and internal boundary metrics include explicit support counts. No
boundary is scored at the first position of an independent scan. Thus shorter
lengths have fewer eligible boundaries, even though page coverage is identical.

The main full-stream overlapping-window evaluation remains the sole checkpoint
selection and early-stopping score. Diagnostics stop between batches after their
fraction of the main evaluation time; one in-flight batch can overrun. Incomplete
levels are marked explicitly, and the evaluation order rotates across validation
steps to avoid starving the same lengths. Durations, coverage and metrics are
logged locally and to ClearML. After training finishes, the selected stage-best
checkpoint receives a larger, uncapped comparison written to
`final-context-evaluation.json` and logged under `final_context`.

## Loader and logging memory

Document-pair coverage is diagnostic only. It uses a fixed 16-KiB HyperLogLog
tracker and is reported as `unique_join_pairs_estimate` (about 0.81% relative
standard error), rather than retaining every encountered Python pair tuple.
It consumes no training RNG and does not influence sampling. Checkpoint sampler
snapshots copy flat bounded containers; loaded checkpoint histories and temporary
models are released after restoration. Volume identities remain exact and are
bounded by the catalogue. The sampler can read legacy exact-pair histories by
streaming them into the fixed-size tracker.

Native resume still checks exact config/data/code/environment identity. The
explicit offline `python -m page_sequence.checkpoint_migration` tool supports only
the audited pre-fix source digest. It backs up original checkpoints, verifies the
precise allowed source changes, compacts diagnostic pair histories, updates source
provenance with a recorded migration, and checks content hashes of all protected
training state before publication. No automatic source-mismatch bypass is used.
Checkpoint RNG is restored after ClearML initialization, which otherwise reseeds
Python/NumPy/Torch and changes the resumed augmentation/dropout sequence.

### Optional per-page document-edge supervision

With auxiliary heads enabled, `loss_weights: {"boundary_consistency": 0.1}`
adds a training-only agreement term in both stages (default zero). For each
valid adjacent pair, it averages Bernoulli Jensen-Shannon divergences between
the boundary-before-current-page probability and (a) current-page start and
(b) previous-page end. Probabilities/log probabilities accumulate in FP32;
gradients reach all three heads and the shared encoder, with no detached teacher.
The raw term is bounded by log(2) and uses duplicate-aware pair weights with its
own denominator. Padding, absent neighbours, unknown auxiliary labels and
conflicting targets are excluded; conflicts are counted explicitly. Existing
supervised losses remain active to anchor predictions. This is an agreement
regularizer, not a guarantee of long-context use. It does not widen the boundary
head, combine auxiliary probabilities at inference, or change evaluation or
checkpoint selection. Effective coefficients and the consistency loss are logged.

Loss weights can be overridden by stage, for example
`stage_loss_weights: {"finetune": {"weak_type": 0.05}}`. Unspecified tasks inherit
`loss_weights` and then the standard defaults. This example keeps unannotated
page-type weight 0.2 in pretraining and uses 0.05 in fine-tuning; annotated type,
side, pair-boundary and auxiliary edge losses keep their original coefficients.
Weights must be finite and nonnegative, and unknown stages/tasks are rejected.
Effective weights are recorded in each metrics row and logged to ClearML, including
on resume. Each task retains its own valid-target denominator. Window 512 is
supported by BERT Mini's existing position table; scanning uses stride 256, and
context diagnostics include a 512-page level when configured.

To isolate a fine-tuning loss change, branch from the baseline's actual
`finetune` / `stage_step: 0` transition checkpoint. This contains the selected
pretraining model/optimizer/scaler plus the final pretraining sampler and RNG
state used to enter fine-tuning; `pretrain_best.pt` alone has a different sampler
position and is not that exact initialization. `fork_finetune.py` explicitly
checks this transition and permits only the weak-type 0.2-to-0.05 fine-tuning
change, new logging/run identity, and content-equivalent local store paths.
It checks code/data/environment/features, verifies protected training state
after serialization, and creates a separate ClearML history with a parent audit.
The ordinary trainer then uses its normal strict resume checks. A watcher can
snapshot this atomic transition checkpoint while the unchanged baseline keeps
training. GPU numerical nondeterminism may still cause later roundoff differences,
but both variants begin fine-tuning from exactly the same stored state.
The watcher supports `--wait-for-status` to capture and prepare the shared state
immediately but queue training behind an occupied SSD. It then replaces itself
with a lightweight slot waiter, releasing its Torch runtime memory. The waiter
requires successful completion and never launches a third simultaneous trainer.

`auxiliary_edges: true` adds independent linear `page_start_head` and
`page_end_head` classifiers on each contextual BERT vector. The existing
previous/current-vector boundary head, its first-position mask, overlap merging,
full-stream metrics and checkpoint-selection criterion are unchanged. Auxiliary
probabilities are not inputs to that boundary head: both auxiliary losses instead
backpropagate into the shared encoder. Default auxiliary loss coefficients are
`page_start: 0.25` and `page_end: 0.25`; existing coefficients stay unchanged.

Targets describe the first/last retained catalogue page of the bottom-level
document. They are not crop boundaries and are not inferred from page types or
page numbers. The native reader derives them from its existing document table,
with no store rebuild or metadata queries per window. SQL/fallback readers use
bounded document-bound queries. Unknown targets and padding are ignored. Known
targets at both window endpoints, including singleton scans, remain supervised.
This does not verify whether a source scan omitted physical pages outside the
catalogue. Targets and document identities never become model input features.

`edge_crops: {"start_probability": 0.1, "end_probability": 0.1}` optionally
crops anchored/interesting windows at a visible catalogue beginning or ending,
removing the neighbouring document on that side. All annotated pages in the
original window must remain. If no eligible edge is visible, the window stays
unchanged. Coverage-source windows are never cropped, so the coverage guarantee
is preserved. These probabilities apply to non-coverage windows (40% of pretrain
and 80% of fine-tune draws). Requested/applied/unavailable crops and actual
window-first/last document-edge exposures are counted. The policy and sampling
RNG are checkpointed; a changed policy fails resume validation.

Existing context diagnostics gain optional auxiliary metrics over all labelled
pages and over scan endpoints. The selected final checkpoint additionally gets
`final-document-edges.json`: a bounded, deterministic dev-only panel of document
prefixes, tails and interior negatives, without neighbouring documents. These
diagnostics are logged under `final_document_edges` and do not select checkpoints.
Portable MetaKat export preserves both auxiliary heads and their specification flag.
MetaKat accepts only the pagination-aware BERT/separate-MLP architecture with
these heads (`metakat_context_v3`). The pair boundary head still supplies the
final boundary prediction; auxiliary probabilities are not combined into it.

For concurrent runs, `worker_start_method: "forkserver"` starts workers from a
separate CPU-only process with preloaded Torch, NumPy and dataset modules. This
shares runtime memory without forking the CUDA-initialized trainer or copying
its populated sampler catalogue. The default remains `spawn`; direct `fork` is
rejected. Prefetch stays bounded to one batch per worker. Inline and forkserver
loading are tested for identical rows, features and supervision.

Set `clearml.report_in_subprocess: false` to use the SDK's thread reporting mode.
This avoids a reporting subprocess retaining the trainer's loaded catalogue. The
override applies only within the experiment process, preserving global settings.
Eight-worker GPU training and validation are covered by the loader tests.

ClearML resume reuses `run_dir/clearml.json` and preserves the existing task
history. Integer `continue_last_task=0` disables the SDK's automatic iteration
offset because the trainer already restores and reports absolute global steps.
A missing task-ID file fails explicitly instead of silently creating a new task.
Checkpoint and log persistence are separate: updates after the last checkpoint
can be replayed at overlapping step numbers, and buffered events may be lost on
a hard kill. No ClearML file server is required; checkpoints remain local.

### Optional pagination experiment

`pagination: {"path": "/absolute/path/pagination.npy", "cap": 10000}` adds a separate
pagination MLP to the existing modality fusion. Each page receives five FP16 features:
`log1p(min(n,cap))/log1p(cap)`, odd parity, Arabic flag, Roman flag, and availability.
The cap is fixed independently of document/window length. Missing pagination is all zero;
the branch output is masked after its biased MLP. Document length, DB page order,
and adjacent-number differences are not added as features.

These are **noisy DB candidates**, not verified physical printing. NDK permits some inferred
or corrected numbers without brackets. The parser accepts positive Arabic integers, canonical
Roman numerals, and the printed component of explicit corrections such as `55 [58]`.
Bracket-only values, suffixes, ranges, multiple-number/spread values, zero, and malformed
values are unavailable in this first experiment. No bracketed content becomes a number.

Build a row-aligned sidecar without changing the native feature store:

```bash
python -m page_sequence.pagination --index /store/catalogue.sqlite \
  --source /metadata/db.256.final.pruned.trn_docs.csv --output /work/pagination.npy
```

The builder streams the CSV, retaining at most 1024 cached document mappings / 262144 cached page identities, plus a bounded
SQLite cache and a row-coverage bitmap. It verifies every retained fit/dev page exactly once;
held-out test pagination is unused. It publishes checksums and coverage in `pagination.json`.
Training/evaluation verify the catalogue identity and sidecar checksum; worker processes open
an mmap. Resume verifies the candidate policy, cap, and data identity. Historical configurations
and checkpoints retain their original three-modality behavior.

For arbitrary-scan inference, `Page.pagenum` / the predict CLI's JSONL `pagenum` is optional.
The same parser and normalizer apply; missing numbers require no imputation.
MetaKat exports use `metakat_context_v3`, preserving the start/end heads and carrying
the candidate policy and fixed cap alongside the model. The consumer reads existing
`MetakatPage.pageNumber.text` and applies the same parser and FP16 encoding. Legacy
v1/v2 artifacts and checkpoints without auxiliary heads are rejected. The artifact
needs neither the sidecar nor the DB dump; `predict`/`tag_pages` also support caller-provided numbers.

Whole-modality robustness training is optional:
`modality_dropout: {"all_probability": 0.70, "two_probability": 0.21,
"single_probability": 0.09}`. Each training window retains all three original
modalities (image, text, statistics), exactly two, or exactly one with these
probabilities. Subsets of each size are uniform: each single-modality case has
3% probability. The selection applies throughout the window, with no new
scattered dropout. If natural missingness would leave an individual page with
no signal, its first originally available modality is retained. Originally
all-missing pages remain missing. Dropped vectors and presence flags are zero;
fusion masks the projected branches after their biased layers. Pagination is
independent. Both training stages use augmentation; validation/inference do not.
Omitting the setting preserves existing behavior and consumes no random numbers.
The checkpointed Torch RNG also records these choices for reproducible resume.

`scattered_dropout: {"modality_probability": 0.10, "page_probability": 0.05}`
runs after the whole-window selection. Each page independently loses each of
image/text/statistics with probability 10%, with no last-modality rescue. A
separate 5% complete-page event zeros every input, including pagination and
availability. Positions, attention masks and training targets stay intact; even
an entirely blank singleton scan remains supervised. Existing DB gaps that were
skipped are not themselves empty model positions. Both probabilities default to
zero, consume no RNG when disabled, and use checkpointed Torch RNG when enabled.

`robustness_evaluation: true` adds final-checkpoint-only diagnostics on the bounded
final dev panel, at the training window length. Six passes remove each original
modality separately or retain only one (pagination remains available in these
ablations). Two more passes compare normal and corrupted inputs on identical
pages, using a fixed local NumPy seed, 10% independent modality failures and 5%
complete-page failures. Catalogue identities receive consistent masks. Results
include separate complete-masked-page and retained-page metrics; boundary scores
are assigned to the current page, excluding each scan's first position as usual.
All reports contain panel/mask identities, coverage and the selected checkpoint
step, and are saved as `final-robustness-*.json` and logged to ClearML under
`final_robustness`. These are enriched diagnostic samples, not corpus estimates;
they do not change recurring validation, checkpoint selection or training RNG.
The same diagnostic suite can be run on trusted existing checkpoints:

```bash
python -m page_sequence.robustness --checkpoint /run/finetune_best.pt \
  --output-dir /work/robustness --device cuda --sample-pages 8192
```

For robustness training, set `pagination_dropout: {"window_probability": 0.2,
"page_probability": 0.2}`. In both pretraining and fine-tuning, 20% of sampled windows
lose all pagination, and each page in the remaining windows independently loses it
with probability 20%. Thus 64% of originally available page signals remain on average.
All five components, including availability, become zero; retained values are never
rescaled. Other modalities and all task targets remain intact. This augmentation runs
only in the training loop, using the checkpointed Torch RNG. Default probabilities are
zero; evaluation and inference use every supplied candidate.

Set `pagination_robustness_evaluation: true` alongside enabled context evaluation to
evaluate the selected final checkpoint a second time on the same bounded development
panel with all pagination removed. Results go to `final-context-without-pagination.json`
and ClearML `final_context_without_pagination`, alongside the normal final diagnostics.
This adds a final diagnostic pass, without adding work to recurring validation or
changing checkpoint selection.
