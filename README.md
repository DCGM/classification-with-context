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

The split per class, with N annotated pages of the class: N ≤ 200 → half in tst, half in trn;
N > 200 → about 100 in tst (up to ~120, as whole documents move), the rest in trn. The tst pages
are spread over as many documents as possible (at most 6 tst pages of one class come from one
document). A document that holds a copy of a tst page (the same page uuid in another library) is a
tst document too.

| File | Rows | Content |
|---|---|---|
| `db.256.final.pruned.trn_docs.csv` | 45,056,755 | trn context: all pages of all documents except the tst documents (annotated or not) |
| `annotated.trn.256.final.csv` | 24,442 (13,146 documents) | the annotated rows of the trn context file |
| `db.256.final.pruned.tst_docs.csv` | 493,714 (2,111 documents) | tst context: all pages of the documents of the tst pages |
| `annotated.tst.256.final.csv` | 2,536 (2,103 documents) | the annotated rows of the tst context file |

The two context files are the whole collection: together they are the pruned DB dump, and each
page is in one of them only. The annotated files are subsets of them: an annotated file is just
the rows of its context file whose annotated page type (column 7) is filled, i.e. a grep of the
larger file: the same lines, only in the annotation order.

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
copy with text is kept. A check of the documents with annotated pages (2026-10-07) found real gaps
(pages missing in the DB or without text) in under 1% of them.

### Annotated vs DB page types

How often the DB (Kramerius) page type differs from the annotated one, per annotated class.
An empty DB type counts as a difference.

**trn** (24,442 pages, 1,887 differ = 7.7%)

| Annotated type | Pages | DB type differs | % | Most common DB types when different |
|---|---:|---:|---:|---|
| CalibrationTable | 3,228 | 0 | 0.0% |  |
| NormalPage | 2,669 | 1,151 | 43.1% | Table 409, Illustration 242, Map 138 |
| TitlePage | 1,727 | 45 | 2.6% | (empty) 27, NormalPage 9, FrontCover 7 |
| FrontCover | 1,688 | 84 | 5.0% | TitlePage 32, (empty) 27, FrontJacket 20 |
| BackCover | 1,668 | 29 | 1.7% | (empty) 27, Jacket 1, TitlePage 1 |
| BackEndSheet | 1,646 | 37 | 2.2% | (empty) 26, Blank 6, BackCover 2 |
| FrontJacket | 1,361 | 4 | 0.3% | Jacket 2, (empty) 2 |
| Map | 1,335 | 60 | 4.5% | FlyLeaf 55, (empty) 4, Illustration 1 |
| Spine | 1,267 | 0 | 0.0% |  |
| FlyLeaf | 1,219 | 5 | 0.4% | Map 3, (empty) 2 |
| TableOfContents | 1,173 | 28 | 2.4% | (empty) 11, ListOfIllustrations 8, BackCover 3 |
| Blank | 986 | 197 | 20.0% | NormalPage 113, Map 39, (empty) 22 |
| SheetMusic | 952 | 31 | 3.3% | NormalPage 31 |
| Index | 845 | 6 | 0.7% | ListOfIllustrations 3, NormalPage 2, (empty) 1 |
| Table | 713 | 20 | 2.8% | NormalPage 11, ListOfTables 3, FlyLeaf 3 |
| ListOfIllustrations | 443 | 6 | 1.4% | ListOfTables 4, ListOfMaps 2 |
| Jacket | 429 | 1 | 0.2% | FrontJacket 1 |
| FrontEndSheet | 277 | 28 | 10.1% | (empty) 25, FrontEndPaper 3 |
| Illustration | 221 | 32 | 14.5% | NormalPage 22, FlyLeaf 4, CalibrationTable 2 |
| Advertisement | 107 | 17 | 15.9% | NormalPage 14, BackCover 1, Jacket 1 |
| Cover | 106 | 34 | 32.1% | FrontCover 32, Jacket 1, TableOfContents 1 |
| ListOfMaps | 60 | 0 | 0.0% |  |
| ListOfTables | 56 | 5 | 8.9% | ListOfIllustrations 2, NormalPage 1, TableOfContents 1 |
| Errata | 55 | 2 | 3.6% | NormalPage 1, FlyLeaf 1 |
| Frontispiece | 52 | 3 | 5.8% | NormalPage 3 |
| BackEndPaper | 43 | 23 | 53.5% | NormalPage 23 |
| Bibliography | 23 | 5 | 21.7% | NormalPage 4, Table 1 |
| Impressum | 23 | 21 | 91.3% | NormalPage 21 |
| Imprimatur | 14 | 0 | 0.0% |  |
| FrontEndPaper | 13 | 0 | 0.0% |  |
| Preface | 12 | 1 | 8.3% | Cover 1 |
| Dedication | 11 | 2 | 18.2% | NormalPage 2 |
| Colophon | 10 | 10 | 100.0% | (empty) 7, NormalPage 3 |
| FragmentsOfBookbinding | 5 | 0 | 0.0% |  |
| Appendix | 3 | 0 | 0.0% |  |
| Abstract | 1 | 0 | 0.0% |  |
| Edge | 1 | 0 | 0.0% |  |

**tst** (2,536 pages, 260 differ = 10.3%)

| Annotated type | Pages | DB type differs | % | Most common DB types when different |
|---|---:|---:|---:|---|
| TableOfContents | 118 | 2 | 1.7% | ListOfIllustrations 2 |
| Blank | 114 | 44 | 38.6% | Map 29, NormalPage 8, FlyLeaf 3 |
| Table | 111 | 0 | 0.0% |  |
| BackEndSheet | 105 | 4 | 3.8% | Jacket 3, Map 1 |
| Index | 103 | 3 | 2.9% | ListOfIllustrations 2, Map 1 |
| Jacket | 103 | 5 | 4.9% | Map 2, Cover 1, ListOfIllustrations 1 |
| Cover | 101 | 19 | 18.8% | FrontCover 19 |
| FlyLeaf | 101 | 1 | 1.0% | (empty) 1 |
| FrontEndSheet | 101 | 13 | 12.9% | FrontEndPaper 9, Jacket 2, Bibliography 1 |
| FrontJacket | 101 | 1 | 1.0% | (empty) 1 |
| Illustration | 101 | 13 | 12.9% | FlyLeaf 8, NormalPage 2, Cover 1 |
| Advertisement | 100 | 2 | 2.0% | Map 1, Jacket 1 |
| BackCover | 100 | 4 | 4.0% | FlyLeaf 2, Cover 1, Advertisement 1 |
| CalibrationTable | 100 | 0 | 0.0% |  |
| FrontCover | 100 | 22 | 22.0% | FrontJacket 13, Cover 4, TitlePage 3 |
| ListOfIllustrations | 100 | 2 | 2.0% | ListOfMaps 2 |
| Map | 100 | 4 | 4.0% | FlyLeaf 3, (empty) 1 |
| NormalPage | 100 | 53 | 53.0% | Table 21, Illustration 10, Map 8 |
| SheetMusic | 100 | 23 | 23.0% | NormalPage 21, FlyLeaf 2 |
| Spine | 100 | 0 | 0.0% |  |
| TitlePage | 100 | 3 | 3.0% | NormalPage 1, Cover 1, FlyLeaf 1 |
| ListOfMaps | 60 | 2 | 3.3% | (empty) 1, ListOfIllustrations 1 |
| ListOfTables | 55 | 3 | 5.5% | ListOfIllustrations 3 |
| Errata | 54 | 0 | 0.0% |  |
| Frontispiece | 52 | 1 | 1.9% | Illustration 1 |
| BackEndPaper | 42 | 14 | 33.3% | NormalPage 13, FlyLeaf 1 |
| Bibliography | 23 | 2 | 8.7% | NormalPage 1, ListOfIllustrations 1 |
| Impressum | 23 | 8 | 34.8% | NormalPage 8 |
| Imprimatur | 14 | 0 | 0.0% |  |
| FrontEndPaper | 13 | 3 | 23.1% | NormalPage 2, Map 1 |
| Preface | 12 | 0 | 0.0% |  |
| Dedication | 10 | 0 | 0.0% |  |
| Colophon | 9 | 9 | 100.0% | (empty) 4, NormalPage 4, FrontEndSheet 1 |
| FragmentsOfBookbinding | 5 | 0 | 0.0% |  |
| Appendix | 3 | 0 | 0.0% |  |
| Abstract | 1 | 0 | 0.0% |  |
| Edge | 1 | 0 | 0.0% |  |

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
| Advertisement | page type | 107 | 100 | 353,062 |
| Appendix | page type | 3 | 3 | 78 |
| BackCover | page type | 1,668 | 100 | 135,341 |
| BackEndPaper | page type | 43 | 42 | 4,774 |
| BackEndSheet | page type | 1,646 | 105 | 133,867 |
| Bibliography | page type | 23 | 23 | 1,154 |
| Blank | page type | 986 | 114 | 283,334 |
| CalibrationTable | no | 3,228 | 100 | 3,392 |
| Colophon | page type | 10 | 9 | 3 |
| Cover | page type | 106 | 101 | 9,898 |
| CustomInclude | no | 0 | 0 | 46 |
| Dedication | page type | 11 | 10 | 1,546 |
| Edge | page type | 1 | 1 | 4,149 |
| Errata | page type | 55 | 54 | 836 |
| FlyLeaf | page type | 1,219 | 101 | 21,344 |
| FragmentsOfBookbinding | no | 5 | 5 | 52 |
| FrontCover | page type | 1,688 | 100 | 137,513 |
| FrontEndPaper | page type | 13 | 13 | 4,919 |
| FrontEndSheet | page type | 277 | 101 | 137,628 |
| Frontispiece | page type | 52 | 52 | 384 |
| FrontJacket | page type | 1,361 | 101 | 20,655 |
| Illustration | page type | 221 | 101 | 153,239 |
| Impressum | page type | 23 | 23 | 266 |
| Imprimatur | page type | 14 | 14 | 149 |
| Index | page type | 845 | 103 | 282,718 |
| Jacket | page type | 429 | 103 | 156,517 |
| ListOfIllustrations | page type | 443 | 100 | 13,372 |
| ListOfMaps | page type | 60 | 60 | 576 |
| ListOfTables | page type | 56 | 55 | 1,280 |
| Map | page type | 1,335 | 100 | 47,159 |
| NormalPage | page type | 2,669 | 100 | 39,243,261 |
| Obituary | logical part only | 0 | 0 | 3 |
| Preface | page type | 12 | 12 | 2,371 |
| SheetMusic | page type | 952 | 100 | 89,410 |
| Spine | page type | 1,267 | 100 | 4,595 |
| Table | page type | 713 | 111 | 437,761 |
| TableOfContents | page type | 1,173 | 118 | 319,318 |
| TitlePage | page type | 1,727 | 100 | 945,060 |
| **total** | | 24,442 | 2,536 | 42,951,177 |

CustomInclude and Obituary have no annotated pages. Every annotated class has pages in both trn and tst. Two NDK page types are in neither the list nor the data: `afterword` and `conclusion`.

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# gt:   the true class id of each page of the set (from the annotated page type), shape (n_pages,)
# pred: the predicted class id of each page, the argmax of the model's logits, shape (n_pages,)
# label2id: class name -> class id
labels = list(range(len(label2id)))   # all class ids
accuracy = accuracy_score(gt, pred)
# weighted: per-class values averaged with the number of pages of the class as the weight
w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(gt, pred, average='weighted', labels=labels)
# per class: fscore[i] is class id i
precision, recall, fscore, support = precision_recall_fscore_support(gt, pred, average=None, labels=labels)
```

**Why the weighted average.** The tst split is built so the weights mean something (see Data):
every class with enough pages has ~100 tst pages, so these classes count the same, as in a plain
(macro) average; a scarce class is split evenly between trn and tst, so it has fewer tst pages and
counts less, in proportion to how much data there is for it, in tst and in trn alike. This
also damps the noise of the tiny classes, where a single page changes the class F1 a lot. A
class without tst pages has weight 0, so passing all class ids in `labels` changes nothing.
