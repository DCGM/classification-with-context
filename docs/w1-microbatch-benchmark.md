# W1 physical batch benchmark — 2026-11-05

Both trainers were interrupted and subsequently resumed from their checkpoints
to isolate the GPU benchmark. The benchmark used the W1-S recipe, its existing
checkpoint and optimizer state, real NVMe-store inputs, FP16, and eight loaders.
It did not write training checkpoints or report benchmark updates to ClearML.

Each update still contains the same 64 sampled W512 parent windows. Physical
batch size changes only how many single-page chunks enter each forward/backward
call. Task normalization and the single AdamW update per parent batch are retained.
Every measured update includes transfer, augmentation, forward, loss, backward,
gradient unscaling/clipping, AdamW and scaler update. CUDA is synchronized at the
timing boundaries, and losses and gradient norms are checked for finiteness.

## Data already loaded into RAM

Four real parent batches are reused, with two warm-up updates and eight measured
updates for each physical batch size. Weights/optimizer are reset to the same
checkpoint for every setting. These numbers exclude loading but include transfer
and all training work; they are not isolated forward-pass timings.

| Maximum physical pages | Median seconds/update | Peak allocated GPU GiB |
|---:|---:|---:|
| 1,024 | 0.794 | 0.58 |
| 2,048 | 0.412 | 0.76 |
| 4,096 | 0.234 | 1.12 |
| 8,192 | 0.160 | 1.83 |
| 16,384 | 0.142 | 3.24 |
| 32,768 | 0.134 | 5.89 |

The full batch is about 5.9 times faster than 1,024-page splitting in this part
of the test. No setting ran out of GPU memory. The actual batches contain fewer
than 32,768 real pages when their parent windows include padding.

## Including normal loading

The short initial loading test was extended to avoid relying on an initial
prefetch burst. The longer test discards 16 warm-up updates and measures 64
updates for each of the original and fastest physical batch sizes. Both settings
process exactly 1,880,764 pages with matching sampled-row hashes.

| Maximum physical pages | Mean seconds/update | Mean loading wait | Pages/second | Peak GPU GiB |
|---:|---:|---:|---:|---:|
| 1,024 | 1.705 | 0.978 s | 17,237 | 0.58 |
| 32,768 | 1.121 | 0.977 s | 26,218 | 5.90 |

The full batch reduces measured update time by 34.3%, giving 52.1% greater
throughput. Loading wait is essentially unchanged. This is an isolated sample,
not a guarantee of sustained throughput with two concurrent trainings.

For W1, the measured GPU-memory use does not justify splitting into 1,024-page
physical batches. A maximum physical batch of 32,768 is the preferred measured
choice while retaining the same effective optimizer batch. The training recipes
were left unchanged after this benchmark; both interrupted jobs were resumed.

Artifacts:

- `/home/ikohut/tmp/w1-microbatch-benchmark-2026-11-05/results.json`: six cached
  settings and the initial short loading test.
- `/home/ikohut/tmp/w1-microbatch-benchmark-2026-11-05/long-results.json`: extended
  loading measurements, sampling hashes and checkpoint identity.
- `tools/benchmark_context_microbatches.py`: repeatable benchmark, using the
  frozen training source supplied through `PYTHONPATH`.
