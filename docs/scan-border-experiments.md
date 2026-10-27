# Scan-border experiments r14

Prepared on 2026-10-27 from commit `565ec5f`. Both runs use identical, frozen,
checksummed source copies under `/home/ikohut/tmp/scan-borders-r14/`.

| Setting | SSD run | NVMe run |
|---|---|---|
| Window, including border tokens | 256 | 512 |
| Complete-scan page capacity in one window | 254 | 510 |
| Stride | 128 | 256 |
| Batch / workers | 64 / 8 | 64 / 8 |
| Precision | FP16 autocast, FP32 parameters | Same |
| Whole-modality dropout | 70% all / 21% two / 9% one | Same |
| Scattered dropout | 10% per modality/page, 5% complete-page | Same |
| Pagination dropout | 20% window, 20% page | Same |
| Auxiliary start/end heads | Enabled, 0.25 loss each | Same |
| Boundary consistency loss | Disabled | Disabled |
| Weak page-type weight, both stages | 0.2 | 0.2 |
| Training seed | 17 | 17 |
| Pretraining / fine-tuning update budgets | 15,427 / 10,000 | Same |
| Validation intervals | 1,000 pretraining / 250 fine-tuning | Same |
| Stopping | Patience 5; at least one coverage epoch in pretraining | Same |

Learning rates remain encoder 3e-5, new parameters 3e-4, warmup 500 updates,
weight decay 0.01, gradient clipping 1.0. Each run starts from pretrained BERT
Mini, not from a previous context checkpoint. Pagination candidates and all
three modalities come from the existing audited native FP16 data pipeline.

The new sampler constructs complete-document scans before choosing inference
tiles. The 60% long-scan component targets four window lengths (1,024 / 2,048
pages respectively). Other scan-size targets retain the variable-length mixture;
whole documents may exceed a target. Short padded windows always represent a
whole scan with both markers. The selected anchor survives anchored sampling;
coverage iterates every overlapping tile and checkpoints its pending queue.

Validation uses existing full development scan streams, marker-aware tiling and
the same task metrics as earlier runs. Curated page-type scores for document
beginnings, endings and interiors are logged separately. At completion, each
selected fine-tuning checkpoint also produces isolated-document versus continuous
page-type reports and aligned probabilities. Existing cropped diagnostic panels
remain marker-free stress tests, separate from complete-scan evaluation.

## Runtime locations

- Configurations: `/home/ikohut/tmp/scan-borders-r14/w256/config.json` and
  `/home/ikohut/tmp/scan-borders-r14/w512/config.json`.
- Training artifacts: `/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-ssd-b64-w256-scan-borders-scattered-seed17-r14/`
  and `/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-nvme-b64-w512-scan-borders-scattered-seed17-r14/`.
- Services: `page-sequence-w256-scan-borders-r14.service` and
  `page-sequence-w512-scan-borders-r14.service` in the user systemd instance.
- Each runtime root contains `training.log`, `supervisor.log`, `status.json`,
  frozen `code/`, source revision and file checksums.
- Supervisor: `/home/ikohut/tmp/scan-borders-r14/supervise.py`. It stops its run
  if available RAM stays below 2 GiB for ten seconds and never automatically
  retries a failed training process. Explicit resume preserves the ClearML task.

Initial scheduling: the 512 run starts on NVMe immediately. The 256 run waits
automatically for the existing shared-pretraining 0.05 fine-tuning run to release
the SSD slot, using its atomic supervisor status at
`/home/ikohut/tmp/scatter-comparison-w512-r12/weak005-shared/status.json`.
This preserves the older run and limits concurrent long training jobs to two.

[512 ClearML task](http://semant.cz:8090/projects/cf7452f7555d4335a213e1f4159b80a9/tasks/3a70972d058d43a48d77c536dfcd99d9/output/log).
The 256 task is created when its slot opens; its URL will be in the run's
`clearml.json` and runtime `training.log`.

## Validation before launch

190 page-sequence/native-store regression tests passed, including marker
gradients, padding invariance, page/pair coverage, anchor retention, complete
document sampling, worker serialization and exact sampler/checkpoint resume.
Both storage configurations separately passed four real-data CUDA FP16 optimizer
updates with batch 64 and eight workers, plus two development validations, with
finite losses. The two frozen source trees have identical file checksums.
