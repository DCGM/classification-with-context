# Four alternate fine-tuning tails and persistent recovery

Configured 2026-10-28. The four parents are the r15 scan-only 256/SSD and 512/NVMe
experiments and the r16 optional-document-metadata 256/SSD and 512/NVMe experiments.
Each receives a distinct fine-tuning run whose only training change is
`stage_loss_weights.finetune.weak_type = 0.05`, replacing 0.2. Human type, side,
pair boundary and auxiliary start/end losses, data, code, model, windows, loaders,
metadata dropout and other settings stay unchanged. Pretraining is not rerun.

The queue captures each running parent's atomic fine-tuning step-zero checkpoint.
The existing audited fork helper verifies model, optimizer, scaler, RNG, sampler,
feature and data identity, then creates a separate run and ClearML history. Resume
continues that tail's task and absolute iterations. Each child uses its parent's
immutable source copy; the live checkout is not a training dependency.

The already completed r15 256 run did not retain its original transition. Its tail
therefore restores the exact best pretrained model/optimizer/scaler (step 15,000),
sets the actual fine-tuning transition global step to 15,427, and starts fine-tuning
with that saved checkpoint's RNG/sampler. This is not the parent's exact RNG/sampler
continuation. The explicit reconstruction audit is retained beside the snapshot;
no final fine-tuned weights are used to initialize the alternative tail.

## Operation

- Persistent service: `page-sequence-experiment-monitor.service` (enabled).
- User lingering is enabled, so the service survives logout and can start at boot.
  Native disks still need to be mounted before data access can succeed.
- Runtime and frozen monitor helpers: `/home/ikohut/tmp/finetune-tails-r17/`.
- Registry: `monitor-config.json`; live state: `state.json`.
- Event history: `events.jsonl`; daemon log: `monitor.log`.
- Tail runtimes: `r15-w256`, `r15-w512`, `r16-w256`, `r16-w512` under that root.
- Trainer artifacts: the parent run names suffixed `-ftweak005-shared-r17` under
  `/home/ikohut/tmp/page-sequence-runs/`.

A single locked monitor adopts current supervisors and counts all local
`page_sequence.runner train` processes, including unregistered jobs. It starts at
most three trainings total. A checkpoint helper is not a training and only one
helper runs at once. Tails wait for slots; interrupted original parents receive
priority. A launch requires 6 GiB available RAM. Under 3 GiB for ten seconds,
the monitor interrupts its newest managed training first; existing supervisors
retain their own guards. Helper work requires 2.5 GiB available RAM.

Training attempts run in independent systemd units with process-group cleanup and
persistent exit markers. The monitor service restarts if it fails and reconstructs
its queue from disk. Transient/OOM/worker failures resume from `latest.pt` after
exponential delay (one minute to 30 minutes), when memory and a slot permit.
Future tails can also resume their prepared `fork-init.pt` before the first saved
training checkpoint. Config/data/code/environment checks stay strict. No progress
for 30 minutes triggers an interruption and checkpoint resume; checkpoint helpers
also have a 30-minute timeout.

This does not silently retry deterministic errors: identity mismatches, disk-full,
missing Python modules, permission errors and repeated numerical failures require
diagnosis. Three stopped attempts without advancing training also enter
`needs_diagnosis`. The error/log tail is preserved in the runtime's
`last-failure.json`. A parent that has training history but no checkpoint also
requires diagnosis rather than silently replacing its run. Other healthy jobs
continue. State records successful completion only after training and final
reports return exit code zero. The monitor cannot repair arbitrary code or data
errors autonomously; these cases remain visibly unfinished.

## Validation

Ten tests passed for the fork and monitoring helpers, covering shared-state
preservation, first-tail reconstruction, invalid fork rejection, transition
capture, external-job concurrency accounting, retry limits, partial log writes and
exit-status recording. Each parent's frozen source inventory was verified before
copying it to the corresponding tail runtime. No running trainer source or config
was modified.
