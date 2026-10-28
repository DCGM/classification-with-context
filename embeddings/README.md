# Page embeddings

Resumable extractors that write one LMDB of float16 vectors per model, keyed like their source. Each output
directory holds `lmdb/`, `meta.json` (settings a resumed run must repeat), `progress.tsv`, `log.txt` and
`run.lock`. Rerunning the same command resumes; `--n-gpu N` (text) uses several GPUs, `--limit` is for tests.

| Package | Embedding | Source | Run |
|---|---|---|---|
| `text/mmbert` | mmBERT-base, masked mean of the last hidden state (768) | text dump LMDB | `text/run_infer_text_embeddings.sh mmbert <public\|private>` |
| `text/qwen3` | Qwen3-Embedding-0.6B, last token (1024) | text dump LMDB | `text/run_infer_text_embeddings.sh qwen3 <public\|private>` |
| `image/dinov2` | DINOv2 ViT-B/14 with registers, CLS token (768) | `<library>/<document>/<page>.jpg` tree | `python -m embeddings.image.dinov2.infer_dinov2 --image-dir … --checkpoint … --output-root …` |
| `text/page_stats` | page layout statistics | OCR XML in the document zips | see its README |

`ocr_xml.py` reads OCR XML; `copy_lmdb_compact.py` and `run_copy_embeddings.sh` copy finished LMDBs.

## Text

`--normalization` decides what the encoder reads (`text/normalization.py`), applied before the empty-page
check and the `--max-chars` cap (4000, as in MetaKat):

- `ocr_words` (default): the words of the page separated by single spaces. No layout reaches the text
  encoders; line counts, lengths and positions are the page statistics' job, and the encoders are not known
  to read line breaks sensibly. Pages without words all get the embedding of `""`.
- `ocr_lines`: the same with the non-blank lines kept.
- `none`: the dump text unchanged, which is how the first mmBERT and Qwen3 embeddings were computed. With
  it the mmBERT script reproduces the stored vectors (mean cosine 0.99999 on 3000 public pages).

The mode is part of `meta.json`, so a resume cannot mix modes. Write the new embeddings to a new
`--model-name` (the run script uses `<Model>.ocr_words`). MetaKat must give its encoder the same string
(`reading_order_text` joins lines with `\n` today, for the checkpoints trained on `none`), so change both
together with the retraining.

mmBERT runs as MetaKat's `MmbertEncoder` does: float32 weights under bfloat16 autocast, sdpa attention, the
padding-NaN patch, float16 storage.

## Image

The page is reduced to the 256 px thumbnail the original dump used (longest edge 512 px, JPEG Q90, then 256 px,
JPEG Q90; `image/dinov2/thumbnail.py`) and resized to 224x224 with `cv2.INTER_AREA`, as MetaKat's
`DinoEncoder` does, so the input may be the 256 px thumbnails or full-resolution scans.

Tests: `python -m pytest embeddings`. `test_thumbnail.py` also compares with MetaKat's thumbnail when `metakat`
is importable.
