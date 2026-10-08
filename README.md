# classification-with-context

## Where things are

```
/mnt/matylda0/ihradis/digiknihovna.data/
├── page_types_metadata_for_training/       the four CSV files (see Data)
├── public/
│   ├── text/<model>/       text embeddings of the pages in the public text dump
│   └── image/<model>/      image embeddings of the images in public.256
└── private/
    ├── text/<model>/       text embeddings of the pages in the private text dump (none yet)
    └── image/<model>/      image embeddings of the images in private.256
```

Each `<model>` folder holds `lmdb/` and `meta.json`, plus the logs of the run that made it. The
models computed so far are listed in [Page embeddings](#page-embeddings-lmdb).
The page images themselves (the images base directory of the CSV paths) are in
`/homes/ihradis/naki.images/` (`public.256/`, `private.256/`).

`public`/`private` in these folder names is where the data came from (text dump, image root),
not the access of a page; the access is in the CSV (see below).

## Data

Location: `/mnt/matylda0/ihradis/digiknihovna.data/page_types_metadata_for_training/`

Pages from the Kramerius DB dumps, kept only when the page has both an image embedding
(DINOv2) and a text embedding (mmBERT and the other text models). The annotated pages are
split into trn and tst **by document**: no document has pages in both splits.

| File | Rows | Content |
|---|---|---|
| `db.256.final.pruned.trn_docs.csv` | 45,035,974 | trn context: all pages of all documents except the tst documents (annotated or not) |
| `annotated.trn.256.final.csv` | 20,927 (10,669 documents) | the annotated rows of the trn context file |
| `db.256.final.pruned.tst_docs.csv` | 514,495 (2,158 documents) | tst context: all pages of the documents of the tst pages |
| `annotated.tst.256.final.csv` | 2,266 (2,118 documents) | the annotated rows of the tst context file |

The two context files are the whole collection: together they are the pruned DB dump, and each
page is in one of them only. The annotated files are subsets of them: an annotated file is just
the rows of its context file whose annotated page type (column 7) is filled, i.e. a grep of the
larger file: the same lines, only ordered as in the original annotation lists.

**Format:** CSV without a header, 13 columns. A field is quoted only when it contains a
comma (e.g. `"40, 41"`), and empty fields are left empty.

| # | Column | |
|---|---|---|
| 1 | library | e.g. `mzk`, `nkp`, `cuni_fsv` |
| 2 | root uuid | the top-level ancestor (e.g. the periodical); may be empty |
| 3 | parent uuid | the direct parent of the document (e.g. the volume); may be empty |
| 4 | document uuid | the document the page belongs to (book, issue, ...) |
| 5 | page uuid | |
| 6 | DB page type | the page type from Kramerius; empty when the DB has none (the DB's `not_found` is stored as empty, in every column) |
| 7 | annotated page type | the annotated label; empty when the page is not annotated |
| 8 | placement | `left`, `right`, `single` or empty |
| 9 | order | 0-based position of the page in the document |
| 10 | page number | the printed page number, e.g. `12`, `[1a]`, `IV` |
| 11 | date | e.g. `1933-09-01 00:00:00`; may be empty |
| 12 | access | `public` or `private`; the access from the DB, and the only reliable one |
| 13 | image path | relative to the images base directory (the one holding `public.256/` and `private.256/`): `{root}/{library}/{document_dir}.images/{file}.jpg`, e.g. `private.256/knav/51bce29a-….images/cf085b84-….jpg` |

All four files have this format.

The image path in the CSV is where the image is. A note for using the image directories without
the CSV: the file name is always `{page_uuid}.jpg`, but the `.images` directory is not always named
after the document. For ~7.7M records (mostly periodicals in `mzk` and `knav`) it is the root
uuid, so all issues of a periodical share one directory; occasionally it is the parent or another
uuid, or the document uuid in upper case.

The root in the path (and the LMDB a key is found in) does not tell the access: about 2M
images are stored under the other root. Use the access column.

**Order caveats** (rare, mostly `mzk`): some documents number the orders in steps of 2
(0, 2, 4, ... with consecutive page numbers), and some list every page twice in the DB; only the
copy with text is kept. About 80 of the 12,302 documents with annotated pages have real gaps
(pages missing in the DB or without text).

### Annotated vs DB page types

How often the DB (Kramerius) page type differs from the annotated one, per annotated class.
An empty DB type counts as a difference.

**trn** (20,927 pages, 1,900 differ = 9.1%)

| Annotated type | Pages | DB type differs | % | Most common DB types when different |
|---|---:|---:|---:|---|
| NormalPage | 2,656 | 1,117 | 42.1% | Table 405, Illustration 242, Map 125 |
| TitlePage | 1,758 | 46 | 2.6% | (empty) 27, NormalPage 10, FrontCover 7 |
| FrontCover | 1,745 | 99 | 5.7% | TitlePage 33, FrontJacket 32, (empty) 28 |
| BackCover | 1,743 | 29 | 1.7% | (empty) 27, TitlePage 1, Jacket 1 |
| BackEndSheet | 1,710 | 39 | 2.3% | (empty) 26, Blank 6, Jacket 3 |
| FrontJacket | 1,357 | 5 | 0.4% | (empty) 3, Jacket 2 |
| Map | 1,315 | 56 | 4.3% | FlyLeaf 50, (empty) 5, Illustration 1 |
| Spine | 1,265 | 0 | 0.0% |  |
| FlyLeaf | 1,211 | 6 | 0.5% | Map 3, (empty) 3 |
| TableOfContents | 1,186 | 28 | 2.4% | (empty) 11, ListOfIllustrations 8, BackCover 3 |
| Blank | 993 | 204 | 20.5% | NormalPage 119, Map 37, (empty) 22 |
| Index | 838 | 9 | 1.1% | ListOfIllustrations 5, NormalPage 2, (empty) 1 |
| Table | 718 | 17 | 2.4% | NormalPage 10, FlyLeaf 3, Index 1 |
| Jacket | 497 | 1 | 0.2% | FrontJacket 1 |
| CalibrationTable | 486 | 0 | 0.0% |  |
| ListOfIllustrations | 436 | 7 | 1.6% | ListOfTables 4, ListOfMaps 3 |
| FrontEndSheet | 274 | 39 | 14.2% | (empty) 25, FrontEndPaper 12, Jacket 2 |
| Illustration | 217 | 31 | 14.3% | NormalPage 22, FlyLeaf 4, (empty) 2 |
| Advertisement | 109 | 19 | 17.4% | NormalPage 14, Jacket 2, Map 1 |
| Cover | 108 | 2 | 1.9% | TableOfContents 1, Jacket 1 |
| BackEndPaper | 78 | 36 | 46.2% | NormalPage 36 |
| SheetMusic | 53 | 52 | 98.1% | NormalPage 52 |
| ListOfMaps | 40 | 1 | 2.5% | (empty) 1 |
| Impressum | 30 | 29 | 96.7% | NormalPage 29 |
| Bibliography | 27 | 7 | 25.9% | NormalPage 5, Table 1, ListOfIllustrations 1 |
| ListOfTables | 27 | 1 | 3.7% | NormalPage 1 |
| FrontEndPaper | 21 | 2 | 9.5% | NormalPage 2 |
| Colophon | 10 | 10 | 100.0% | (empty) 6, NormalPage 4 |
| Errata | 9 | 2 | 22.2% | FlyLeaf 1, NormalPage 1 |
| Frontispiece | 4 | 4 | 100.0% | NormalPage 3, Illustration 1 |
| Dedication | 3 | 2 | 66.7% | NormalPage 2 |
| Preface | 2 | 0 | 0.0% |  |
| Abstract | 1 | 0 | 0.0% |  |

**tst** (2,266 pages, 245 differ = 10.8%)

| Annotated type | Pages | DB type differs | % | Most common DB types when different |
|---|---:|---:|---:|---|
| Map | 120 | 8 | 6.7% | FlyLeaf 8 |
| NormalPage | 113 | 87 | 77.0% | Table 25, Map 21, Illustration 10 |
| Index | 110 | 0 | 0.0% |  |
| FlyLeaf | 109 | 0 | 0.0% |  |
| Blank | 107 | 37 | 34.6% | Map 31, FlyLeaf 4, NormalPage 2 |
| ListOfIllustrations | 107 | 1 | 0.9% | ListOfMaps 1 |
| Table | 106 | 3 | 2.8% | ListOfTables 2, NormalPage 1 |
| FrontJacket | 105 | 0 | 0.0% |  |
| TableOfContents | 105 | 2 | 1.9% | ListOfIllustrations 2 |
| FrontEndSheet | 104 | 2 | 1.9% | Map 1, Bibliography 1 |
| Illustration | 103 | 12 | 11.7% | FlyLeaf 8, NormalPage 2, Cover 1 |
| Spine | 102 | 0 | 0.0% |  |
| Errata | 100 | 0 | 0.0% |  |
| Frontispiece | 100 | 0 | 0.0% |  |
| Cover | 99 | 51 | 51.5% | FrontCover 51 |
| Advertisement | 98 | 0 | 0.0% |  |
| ListOfTables | 84 | 7 | 8.3% | ListOfIllustrations 5, Index 1, TableOfContents 1 |
| ListOfMaps | 80 | 1 | 1.2% | ListOfIllustrations 1 |
| TitlePage | 69 | 2 | 2.9% | Cover 1, FlyLeaf 1 |
| CalibrationTable | 56 | 0 | 0.0% |  |
| FrontCover | 43 | 7 | 16.3% | Cover 3, TitlePage 2, FlyLeaf 1 |
| BackEndSheet | 41 | 2 | 4.9% | Map 1, Jacket 1 |
| Jacket | 35 | 5 | 14.3% | Map 2, FlyLeaf 1, ListOfIllustrations 1 |
| Imprimatur | 28 | 0 | 0.0% |  |
| BackCover | 25 | 4 | 16.0% | FlyLeaf 2, Cover 1, Advertisement 1 |
| Preface | 22 | 1 | 4.5% | Cover 1 |
| Bibliography | 19 | 0 | 0.0% |  |
| Dedication | 18 | 0 | 0.0% |  |
| Impressum | 16 | 0 | 0.0% |  |
| FragmentsOfBookbinding | 10 | 0 | 0.0% |  |
| Colophon | 9 | 9 | 100.0% | (empty) 5, NormalPage 3, FrontEndSheet 1 |
| BackEndPaper | 7 | 1 | 14.3% | FlyLeaf 1 |
| Appendix | 6 | 0 | 0.0% |  |
| FrontEndPaper | 5 | 1 | 20.0% | Map 1 |
| Edge | 2 | 0 | 0.0% |  |
| SheetMusic | 2 | 2 | 100.0% | FlyLeaf 2 |
| Abstract | 1 | 0 | 0.0% |  |

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

Folders: `digiknihovna.data/public/text/<model>/lmdb` and `digiknihovna.data/private/text/<model>/lmdb`

| | |
|---|---|
| Key | `{library}_{page_uuid}` (UTF-8), e.g. `mzk_d24dbf43-9a8a-449a-aa6c-5bd56ed0b21b`; the same keys as the page-text LMDB they were computed from |
| Value | float16 vector |

| Model | Dim | Pooling |
|---|---|---|
| `mmBERT-base` | 768 | mean of the last hidden state |
| `Qwen3-Embedding-0.6B` | 1024 | last token |

### Image embeddings

Folders: `digiknihovna.data/public/image/<model>/lmdb` (the images of `public.256`) and
`digiknihovna.data/private/image/<model>/lmdb` (the images of `private.256`)

| | |
|---|---|
| Key | the image path relative to its root (`public.256/` or `private.256/`), UTF-8: `{library}/{document_dir}.images/{page_uuid}.jpg` (`document_dir` is usually the document uuid; see the note in the Data section) |
| Value | float16 vector |

| Model | Dim | Pooling |
|---|---|---|
| `vitb14_dinov2_2026-09-20_public_private_2` | 768 | CLS token after the final LayerNorm; DINOv2 ViT-B/14 with registers, fine-tuned with LightlyTrain |

A page shared by two documents is stored under both document paths.

### Finding the embeddings of a CSV record

For a record of the Data CSV files:
- text: key `f'{library}_{page_uuid}'` in `public/text/<model>/lmdb` or `private/text/<model>/lmdb`
  (a page is in one of them)
- image file: `os.path.join('/homes/ihradis/naki.images', path)`
- image embedding: the path's first component (`public.256` / `private.256`) selects
  `public/image/<model>/lmdb` or `private/image/<model>/lmdb`, and the rest of the path,
  `{library}/{document_dir}.images/{page_uuid}.jpg`, is the key.

## Page types

The label space of the future classifier: the 38 classes of MetaKat's page type classifier
(`page_type_classes` in `metakat/page_type/datasets_from_mods/mods_helper.py`) plus **Colophon**,
added from the NDK digitisation standards (`ppp_mono_2.4`, `ppp_perio_8.7`, table 1.2.2). Names
are matched case-insensitively (the standards write them in camelCase, e.g. `flyleaf`). The NDK
column says whether the standards list the type as a page type, only as a logical part, or not at
all. Counts: annotated pages in trn and tst, and DB (Kramerius) page type over the whole collection
(both context files, 45,550,469 pages):

| Page type | NDK | trn | tst | DB pages |
|---|---|---:|---:|---:|
| Abstract | logical part only | 1 | 1 | 147 |
| Advertisement | page type | 109 | 98 | 353,062 |
| Appendix | page type | 0 | 6 | 78 |
| BackCover | page type | 1,743 | 25 | 135,341 |
| BackEndPaper | page type | 78 | 7 | 4,774 |
| BackEndSheet | page type | 1,710 | 41 | 133,867 |
| Bibliography | page type | 27 | 19 | 1,154 |
| Blank | page type | 993 | 107 | 283,334 |
| CalibrationTable | no | 486 | 56 | 3,392 |
| Colophon | page type | 10 | 9 | 3 |
| Cover | page type | 108 | 99 | 9,898 |
| CustomInclude | no | 0 | 0 | 46 |
| Dedication | page type | 3 | 18 | 1,546 |
| Edge | page type | 0 | 2 | 4,149 |
| Errata | page type | 9 | 100 | 836 |
| FlyLeaf | page type | 1,211 | 109 | 21,344 |
| FragmentsOfBookbinding | no | 0 | 10 | 52 |
| FrontCover | page type | 1,745 | 43 | 137,513 |
| FrontEndPaper | page type | 21 | 5 | 4,919 |
| FrontEndSheet | page type | 274 | 104 | 137,628 |
| Frontispiece | page type | 4 | 100 | 384 |
| FrontJacket | page type | 1,357 | 105 | 20,655 |
| Illustration | page type | 217 | 103 | 153,239 |
| Impressum | page type | 30 | 16 | 266 |
| Imprimatur | page type | 0 | 28 | 149 |
| Index | page type | 838 | 110 | 282,718 |
| Jacket | page type | 497 | 35 | 156,517 |
| ListOfIllustrations | page type | 436 | 107 | 13,372 |
| ListOfMaps | page type | 40 | 80 | 576 |
| ListOfTables | page type | 27 | 84 | 1,280 |
| Map | page type | 1,315 | 120 | 47,159 |
| NormalPage | page type | 2,656 | 113 | 39,243,261 |
| Obituary | logical part only | 0 | 0 | 3 |
| Preface | page type | 2 | 22 | 2,371 |
| SheetMusic | page type | 53 | 2 | 89,410 |
| Spine | page type | 1,265 | 102 | 4,595 |
| Table | page type | 718 | 106 | 437,761 |
| TableOfContents | page type | 1,186 | 105 | 319,318 |
| TitlePage | page type | 1,758 | 69 | 945,060 |
| **total** | | 20,927 | 2,266 | 42,951,177 |

CustomInclude and Obituary have no annotated pages. Appendix, Edge, FragmentsOfBookbinding and Imprimatur are annotated only in tst. Two NDK page types are in neither the list nor the data: `afterword` and `conclusion`.

### Extra page types in the DB dumps

The DB page type (column 6) also holds values outside the list above. They are kept in the files
as they are, so a training data loader should decide what to do with them: ignore them as labels
(treat them as unknown, like an empty type) or map them explicitly (e.g. `introduction` to Preface,
`listOfSupplements` to a list type).

| DB page type | NDK | DB pages |
|---|---|---:|
| (empty: no type in the DB) |  | 2,598,830 |
| `introduction` | page type | 251 |
| `manuscriptNotes` | no | 175 |
| `imgDisc` | no | 13 |
| `listOfSupplements` | no | 10 |
| `review` | logical part only | 7 |
| `bibliographicalPortrait` | logical part only | 4 |
| `ListOfSupplements` | no | 2 |
| **total** | | 2,599,292 |

## Page type classification evaluation

What the previous, image-only page type classifier reported. A starting point for evaluating
training in this repo. Code: `metakat/page_type/nets/page_type_evaluator.py` in MetaKat.

At every evaluation step (every 500 training steps), on the whole evaluation set (here: the tst
split), and on 500 training pages (once with and once without augmentation, to watch
overfitting), it reported:

- `loss`: the mean cross-entropy loss
- `accuracy`
- `weighted_precision`, `weighted_recall`, `weighted_fscore`: per-class values averaged with
  the number of pages of the class as the weight
- `precision_<class>`, `recall_<class>`, `fscore_<class>`, `support_<class>`: for every class

All of them come from scikit-learn, over the predictions (argmax of the logits) of the whole set:

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# gt:   the true class id of each page of the set (from the annotated page type), shape (n_pages,)
# pred: the predicted class id of each page, the argmax of the model's logits, shape (n_pages,)
# label2id: class name -> class id
labels = list(range(len(label2id)))   # all class ids, also those missing in the set
present = np.unique(gt)               # the class ids that have pages in the set
accuracy = accuracy_score(gt, pred)
# weighted: classes weighted by their number of pages
w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(gt, pred, average='weighted', labels=present)
# macro: every class weighted the same
m_precision, m_recall, m_fscore, _ = precision_recall_fscore_support(gt, pred, average='macro', labels=present)
# per class, for all class ids: fscore[i] is class id i
precision, recall, fscore, support = precision_recall_fscore_support(gt, pred, average=None, labels=labels)
```
