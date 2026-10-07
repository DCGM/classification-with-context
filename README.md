# classification-with-context

## Data

Location: `/mnt/matylda0/ihradis/digiknihovna.data/page_types_metadata_for_training/`

Pages from the Kramerius DB dumps, kept only when the page has both an image embedding
(DINOv2) and a text embedding (mmBERT and the other text models). The annotated pages are
split into trn and tst **by document**: no document has pages in both splits.

| File | Rows | Content |
|---|---|---|
| `annotated.trn.256.final.csv` | 20,431 (10,193 documents) | annotated trn pages |
| `annotated.tst.256.final.csv` | 2,256 (2,109 documents) | annotated tst pages |
| `db.256.final.pruned.trn_docs.csv` | 45,038,332 | trn context: all pages of all documents except the tst documents (annotated or not) |
| `db.256.final.pruned.tst_docs.csv` | 512,137 (2,149 documents) | tst context: all pages of the documents of the tst pages |

The two context files together are the whole pruned DB dump. Each page appears in one of them only.

**Format:** CSV without a header, 13 columns. A field is quoted only when it contains a
comma (e.g. `"40, 41"`), and empty fields are left empty.

| # | Column | |
|---|---|---|
| 1 | library | e.g. `mzk`, `nkp`, `cuni_fsv` |
| 2 | root uuid | the top-level ancestor (e.g. the periodical); may be empty |
| 3 | parent uuid | the direct parent of the document (e.g. the volume); may be empty |
| 4 | document uuid | the document the page belongs to (book, issue, ...) |
| 5 | page uuid | |
| 6 | page type | annotated files: **the annotated label**; context files: the page type from Kramerius (it can also be `not_found` or empty) |
| 7 | placement | `left`, `right`, `single` or empty |
| 8 | order | 0-based position of the page in the document |
| 9 | page number | the printed page number, e.g. `12`, `[1a]`, `IV` |
| 10 | date | e.g. `1933-09-01 00:00:00`; may be empty |
| 11 | access | `public` or `private`; the access from the DB, and the only reliable one |
| 12 | image path | `/homes/ikohut/naki.images/<root>/{library}/{document_dir}.images/{file}.jpg` |
| 13 | annotated | `true` for an annotated page. All rows are `true` in the annotated files; in a context file, the annotated pages of that split are `true` |

In the context files the page type is the Kramerius one, even for annotated pages. It
differs from the annotated label on ~9% of the trn and ~10% of the tst annotated pages. Take
the label from the annotated file (match by library and page uuid).

**Finding the embeddings of a record:**
- text: key `f'{library}_{page_uuid}'` in the public or the private text LMDB
  (`2026-09-16.db_text_dump` / `2026-09-16.db_text_dump.private`)
- image: key = the image path after `naki.images/<root>/`, i.e. `{library}/{document_dir}.images/{file}.jpg`.
  The file name may be `uuid:{page_uuid}.jpg`, and the document directory may differ in letter case
  from the uuid, so use the path rather than rebuilding it. Look it up in the image embeddings of
  `public.256` and of `private.256`.

The `<root>` in the path and the LMDB a key is found in do not tell the access: about 2M records
are stored under the other root. Use the access column.

**Order caveats** (rare, mostly `mzk`): some documents number the orders in steps of 2
(0, 2, 4, ... with consecutive page numbers), and some list every page twice in the DB; only the
copy with text is kept. About 80 of the 12,302 documents with annotated pages have real gaps
(pages missing in the DB or without text).

## Page embeddings (LMDB)

Each embedding folder holds one LMDB with one vector per page, plus a `meta.json`
describing it (model, pooling, `dtype`, `dim`, `key_format`, `value_format`).

**Value** (same for text and image): the raw bytes of a single little-endian vector,
no header. Decode it with the `dtype` from `meta.json`:

```python
import json, lmdb, numpy as np

meta = json.load(open(f'{folder}/meta.json'))
env = lmdb.open(f'{folder}/lmdb', readonly=True, lock=False)
with env.begin() as txn:
    vector = np.frombuffer(txn.get(key.encode('utf-8')), dtype=meta['dtype'])  # shape (meta['dim'],)
```

Vectors are not normalized (`"normalized": false`).

### Text embeddings

Folders: `2026-09-16.db_text_dump/embeddings/<model>/lmdb`

| | |
|---|---|
| Key | `{library}_{page_uuid}` (UTF-8), e.g. `mzk_d24dbf43-9a8a-449a-aa6c-5bd56ed0b21b`; the same keys as the page-text LMDB `2026-09-16.db_text_dump/lmdb` |
| Value | float16 vector |

| Model | Dim | Pooling |
|---|---|---|
| `mmBERT-base` | 768 | mean of the last hidden state |
| `Qwen3-Embedding-0.6B` | 1024 | last token |

### Image embeddings

Folder: `2026-09-16.db_image_dump/embeddings/vitb14_dinov2_2026-09-18_public/lmdb`
(DINOv2 ViT-B/14 with registers, fine-tuned with LightlyTrain)

| | |
|---|---|
| Key | the image path relative to the image directory (UTF-8): `{library}/{document_uuid}.images/{page_uuid}.jpg` |
| Value | float16 vector |

| Model | Dim | Pooling |
|---|---|---|
| `vitb14_dinov2_2026-09-18_public` | 768 | CLS token after the final LayerNorm |

A page shared by two documents is stored under both document paths. To find the
image embedding of a text key, build the path from the library, document and page
uuid, e.g. from the eval sample columns: `f'{doc}.images/{page_id}.jpg'` where
`doc` is `{library}/{document_uuid}`.
