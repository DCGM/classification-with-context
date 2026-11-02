# Additive finer statistics groups: r19 W512 / SSD

Repeats the fresh-training [r18 recipe](mmbert-ocr-lines-r18.md), using the same
new OCR-lines mmBERT embeddings, with an expanded statistics input. The current
147 normalized features remain a prefix; eleven finer semantic groups append
whole-page/header/footer features, giving 180 values. No language profile is added.
The exact stem lists are in
[`fine_groups.py`](../embeddings/text/page_stats/fine_groups.py).

Normalization uses the existing fixed keyword-count transforms and binary flags.
The complete stats block is divided by `sqrt(180)`, following the existing
dimension-dependent rule. Before block scaling the old prefix is exactly
unchanged; afterward its values are multiplied by `sqrt(147/180)`, about 0.904.
This is a fresh pretrained BERT Mini run, not a zero-initialized warm extension
of an existing context checkpoint. Its stats projection learns 180 inputs.

The SSD raw-sparse store supplies the new view in workers, without changing or
rebuilding any stored vectors. Legacy dense-view stores are rejected for this
configuration because they do not retain the necessary original counts.
The model/feature schema records the exact added groups, dimensions and scaling.

All other training settings match r18: window 512, batch 64, eight loaders,
seed 17, FP16 compute, optional complete document metadata on 50% of scans,
pagination/whole-modality/scattered dropout, auxiliary start/end heads,
pretraining weak-type weight 0.2 and fine-tuning weight 0.05. Update budgets,
learning rates, early stopping, validation panel and the original frozen
validation merging behavior remain unchanged. Pretraining-end, fine-tuning-init
and fine-tuning-end full checkpoints are retained.

Runtime: `/home/ikohut/tmp/stats-fine-groups-r19/w512/`.

Run name:
`bert-mini-fp16-ssd-b64-w512-boundary5-metadata50-scattered-mmbert-ocr-lines-stats180-ftweak005-seed17-r19`.

Store:
`/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/document-store-ocr-lines-sparse-v2/`.

The enabled persistent `page-sequence-r19-start.service` waits for 9 GiB available
RAM before its full-batch smoke test and training launch. This leaves headroom
for the second trainer while the r18 NVMe run continues. GPU capacity and the
two-training concurrency limit are unchanged. The monitor resumes transient
training failures with the same ClearML task; repeated/deterministic errors stop
for diagnosis. Neither the existing run nor its frozen source is modified.

At setup, the new experiment was waiting for memory, not training. Status is in
`start-status.json`, then `state.json`; smoke and full logs are `smoke.log` and
`managed-training.log`. The gate runs four pretraining and two fine-tuning
updates at batch 64/eight loaders before full training and checks the 180-input
schema, finite losses/weights, phase checkpoints, both metadata modes and zero
LMDB fallback. ClearML is disabled for smoke and enabled for the full run.

Validation: 294 regression tests passed across page-sequence, page-statistics and
native-store code. Tests cover exact original-prefix preservation, new group
counts, normalization serialization, worker serialization and equivalence of the
180-feature native/source views. A CPU forward pass using the actual SSD store
and the frozen experiment's pretrained BERT Mini passed: eight pages, 17 nonzero
additional features, all model outputs finite. This preflight is separate from
the pending full-batch CUDA smoke test. Configuration differences, code hashes,
baseline identity and preflight report are retained in the runtime directory.
