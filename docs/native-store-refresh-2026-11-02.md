# New mmBERT / sparse-statistics store placement

Requested final physical placement:

| Store | NVMe | SSD |
|---|---|---|
| Original mmBERT, grouped statistics (V1) | Link only | Retained |
| New `mmBERT-base.ocr_lines`, raw sparse statistics (V2) | New build | Verified copy |

Old SSD source paths remain usable as links into kolosus. All source text/image
embedding and statistics LMDBs are archived there. The remaining private text
LMDB is relocated as an opaque binary directory; its text is never inspected.
The user additionally approved archiving `page-context-dump-r16-w512-v1`
(full-dump predictions/contextual embeddings), with an SSD link retained.

Runtime journals, logs, source configurations and frozen code snapshots:
`/home/ikohut/tmp/native-store-refresh-2026-11-02/`.

The verified relocation order is new private mmBERT, private raw-text LMDB,
full-dump inference output, unused old public stats run, then the old native
store from NVMe to SSD. Existing kolosus copies are reused only after SHA256
comparison. The old NVMe path becomes a link, freeing space for the build.

Final dataset paths:

- old: `/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/document-store-fp16-v1`
- new NVMe: `/home/ikohut/tmp/document-store-ocr-lines-sparse-v2`
- new SSD: `/run/media/ikohut/3de48e10-fa2f-4c8b-bea2-166e76ea8e4a/data/document-store-ocr-lines-sparse-v2`

Monitor `relocation/status.json`, `relocation.log`, `pipeline-status.json`,
`pipeline.log`, and the new NVMe store's `status.json`. A failure is recorded
in a separate failure journal and stops the workflow; diagnose it before
resuming. This file describes the workflow, not a claim that it has finished.

No current training run or frozen experiment code is modified. The original
store remains available for reproducing prior experiments; the refreshed
store has a different audited build identity. The catalogue, sampling
hierarchy, exclusions, labels, logical page rows and pagination are preserved.
