# Five-bit scan-input experiments r15

Prepared on 2026-10-27 from commit `0edc372`. These are fresh 256/SSD and
512/NVMe runs, replacing the stopped r14 scan-token experiments. Their frozen
source copies are identical and checksummed under
`/home/ikohut/tmp/boundary-features-r15/{w256,w512}/code/`.

The boundary branch receives five binary values in this fixed order:

```text
[scan_start, scan_end, metadata_provided, document_start, document_end]
```

For both experiments, the last three values are **always zero**, during training
and evaluation. True catalogue document boundaries remain targets only. The
first/last page of a supplied complete-document scan gets its scan flag;
internal window cuts get no flag. A separate MLP projects this branch alongside
image, text, ALTO statistics and pagination. Its unknown/all-zero state remains
active, and modality/scattered dropout does not alter the boundary inputs.

There are no special input tokens. Windows hold 256 and 512 actual pages,
respectively. Complete-document scan construction, source quotas, anchor
selection, overlapping inference, existing heads and loss weights otherwise
match r14. Short padded windows must be complete scans. The long-scan component
targets four window lengths; whole documents may exceed a scan-size target.

| Setting | Both runs |
|---|---|
| Training batch / loaders | 64 / 8 |
| Precision | FP16 autocast with FP32 parameters |
| Initialization / seed | Pretrained BERT Mini / 17 |
| Fusion | Separate modality MLPs, including the five-bit branch |
| Whole-modality dropout | 70% all / 21% two / 9% one |
| Scattered dropout | 10% per modality/page, 5% whole-page |
| Pagination dropout | 20% window, 20% page |
| Auxiliary start/end losses | 0.25 each |
| Weak type weight | 0.2 in both stages |
| Boundary consistency loss | Disabled |
| Encoder / new-parameter learning rates | 3e-5 / 3e-4 |
| Pretraining / fine-tuning update budgets | 15,427 / 10,000 |
| Validation intervals | 1,000 / 250 updates |
| Stopping | Patience 5; minimum one coverage epoch before pretraining plateau stop |

Inference uses stride 128/ramp 64 for 256 and stride 256/ramp 128 for 512.
Checkpoint selection remains curated development page-type weighted F1.
Separate beginning/ending/interior metrics and final isolated-document versus
continuous comparisons are retained, including aligned per-page probabilities.
The held-out test set remains unused.

## Runtime

- Configurations, logs and status: `/home/ikohut/tmp/boundary-features-r15/w256/`
  and `/home/ikohut/tmp/boundary-features-r15/w512/`.
- Training artifacts: `/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-ssd-b64-w256-boundary5-scan-only-scattered-seed17-r15/`
  and `/home/ikohut/tmp/page-sequence-runs/bert-mini-fp16-nvme-b64-w512-boundary5-scan-only-scattered-seed17-r15/`.
- User services: `page-sequence-w256-boundary5-r15.service` and
  `page-sequence-w512-boundary5-r15.service`.
- Supervisor: `/home/ikohut/tmp/boundary-features-r15/supervise.py`.
  Available RAM below 2 GiB for ten seconds stops its job; failures are not
  automatically retried. Explicit resume uses the frozen code and original
  ClearML task. The r14 services remain stopped.

## Validation

195 regression tests passed across page-sequence and native-store tests.
Both storage configurations separately passed four real-data CUDA FP16 optimizer
updates at batch 64 with eight workers, plus two small development validations.
All saved model parameters were finite; both gradient scalers stayed at 1024.
The smoke specifications contain `boundaries: 5` and no scan-token parameters.

ClearML tasks:

- [256 / SSD](http://semant.cz:8090/projects/cf7452f7555d4335a213e1f4159b80a9/tasks/ff868675c058425ea19759fcf3259a53/output/log).
- [512 / NVMe](http://semant.cz:8090/projects/cf7452f7555d4335a213e1f4159b80a9/tasks/3a7586219ec24d31afdd803bacaeea74/output/log).

Both full runs were verified advancing real optimizer updates with finite losses,
gradient norms and a loss scale of 1024. Runtime launch verification is saved in
`/home/ikohut/tmp/boundary-features-r15/launch-verification.json`.
