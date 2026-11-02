# New mmBERT comparison: r18 W512

Repeats the selected r16 W512 optional-document-metadata recipe and its r17
weak-type-0.05 fine-tuning tail, starting from fresh pretrained BERT Mini.
The previous context-trained model is not the initialization. The intended
experimental change is replacing the old mmBERT vectors with the completed
public/private `mmBERT-base.ocr_lines` dumps.

The new native store retains the same catalogue, document order, DINO vectors,
labels and pagination. It preserves raw sparse statistics and computes the
same grouped, normalized 147-dimensional view in loading workers; the store
integration tests check feature equivalence and exact raw-byte preservation.

| Setting | Value |
|---|---|
| Context / batch / loaders | 512 / 64 / 8 |
| Seed / compute | 17 / FP16 autocast, FP32 parameters |
| Backbone / fusion | Pretrained BERT Mini, separate modality MLPs |
| Document metadata | Complete metadata on 50% of constructed scans |
| Heads | Type, side, pair boundary, auxiliary start and end |
| Whole-modality dropout | 70% all three / 21% two / 9% one |
| Scattered dropout | 10% modality/page, 5% whole page |
| Pagination dropout | 20% window, 20% page |
| Weak page-type loss | 0.2 pretraining, 0.05 fine-tuning |
| Encoder / new-parameter LR | 3e-5 / 3e-4 |
| Update budgets | 15,427 pretraining, 10,000 fine-tuning |
| Validation intervals | 1,000 / 250 updates |
| Early stopping | Patience 5; pretraining requires one coverage epoch |

Checkpoint selection uses the original curated development weighted F1. To keep
the comparison controlled, the frozen original inference/evaluation modules are
retained, including the original stride 256, edge ramp 128, linear weighting
with a 0.1 floor, and anchored final window. These are training validation
settings; they do not change the current MetaKat deployment merging policy.

The original frozen training modules are retained. The only runner patch adds
durable `pretrain_final.pt`, `finetune_init.pt` and `finetune_final.pt` snapshots,
including optimizer, scaler, sampler and RNG state. The current native-store
reader adds support for the new sparse-statistics format. Frozen source hashes,
the runner patch and the configuration differences are saved beside the runtime.

Runtime: `/home/ikohut/tmp/mmbert-ocr-lines-r18/w512/`.

Training artifacts:
`/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-nvme-b64-w512-boundary5-metadata50-scattered-mmbert-ocr-lines-ftweak005-seed17-r18/`.

Data: `/home/ikohut/tmp/document-store-ocr-lines-sparse-v2/`.
Source LMDB paths are provenance; native training has no LMDB fallback.

The persistent `page-sequence-r18-start.service` waits for the new NVMe store's
published manifest and completed validation. Any preparation failure journal
stops launch. It first runs a separate real-data smoke experiment with batch 64,
eight loaders, four pretraining updates and two fine-tuning updates; ClearML is
disabled for this smoke test. Verification requires finite parameters/losses,
zero fallback reads, correct stage weights and durable phase checkpoints.
Only then does the monitor start the full run with ClearML enabled.

The existing monitor implementation manages training attempts, caps total
concurrent trainings at two, resumes transient interruptions from checkpoints,
and stops repeated or deterministic failures for diagnosis. The smoke gate does
not automatically retry failed smoke training. Status and logs are in the runtime:
`start-status.json`, `smoke.log`, `smoke-passed.json`, `state.json`,
`managed-training.log` and `events.jsonl`. A queued experiment is not yet training.
