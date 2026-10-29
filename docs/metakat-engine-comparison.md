# MetaKat engine comparison: chapter-v1.6.0 vs chapter-v1.7.0

**chapter-v1.6.0** predicts the page type of each page alone and has no side or document-start model of its own. **chapter-v1.7.0** reads the whole batch as one scan (512-page windows) and predicts page type, side and document starts together. Both ran over the same prepared batches, with the same images and ALTO.

## What was run

- **chapter-v1.6.0** page types and sides: `metakat.old-page-number.json` of its results, the run with the MetaKat code from before the page-number changes. Its document starts: `metakat.json`, the run with the newer code (the boundary logic did not change).
- **chapter-v1.7.0**: the final run with the current MetaKat code: the page-number engine runs before and after the page-type stage, the interactive PDF has uniform pages. Results of earlier runs of this engine are kept for internal comparison only and are not in this report.

## Reference labels

- **Page type:** the curated (annotated) label where the page has one, otherwise the DB page type. DB types are partly weak labels, so absolute scores understate both engines; the curated rows are the cleanest.
- **Side:** the DB placement (`left`, `right`, `single`), which exists for few pages only.
- **Document starts:** the true document boundaries of each batch (known from the DB hierarchy). The first page of a batch is not predicted and not scored.
- **Sets:** `evaluation` documents come from the dev split, so its results are held-out. `consistency` documents come from the fit split, which the context model may have been trained on: its chapter-v1.7.0 numbers are inflated and shown only as a sanity check.
- Percentages everywhere; *fixed* = pages the second engine gets right and the first gets wrong, *broken* = the reverse (for the page type or side of the reference class).

## Summary (evaluation set, held-out)

- Page type accuracy over 6033 pages: 77.6 for chapter-v1.6.0, 89.8 for chapter-v1.7.0; on the curated pages 76.8 and 92.1.
- Document starts: F1 51.2 for chapter-v1.6.0, 92.9 for chapter-v1.7.0 (222 of 245 starts found).
- Side (108 pages with a DB placement): 76.9 for chapter-v1.7.0; chapter-v1.6.0 has none.
- Page types with at least 20 reference pages where chapter-v1.7.0 has a lower F1 than chapter-v1.6.0: calibrationTable (95.2 → 58.9), cover (95.2 → 0.0).

## Evaluation set (held-out, the clean comparison)

9169 pages in 59 batches; 3136 pages have no reference page type and are not scored for it.

### Page type, all pages together

| | Pages | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| chapter-v1.6.0 | 6033 | 77.6 | 81.0 | 47.0 |
| chapter-v1.7.0 | 6033 | 89.8 | 89.9 | 54.2 |

chapter-v1.7.0 fixes 961 pages and breaks 225 (paired McNemar p = 1.2e-108).
On the 203 curated pages alone: chapter-v1.6.0 accuracy 76.8, chapter-v1.7.0 92.1 (fixed 35, broken 4).

#### Per page type

Precision (P), recall (R) and F1 per engine. Two views of the change, which together explain P and R: *fixed*/*broken* count the pages **of that reference class** that chapter-v1.7.0 gets right or wrong where chapter-v1.6.0 did not or did (the change in recall); *alarms removed*/*added* count the pages **of other classes** that chapter-v1.6.0 wrongly labelled as this type and chapter-v1.7.0 no longer does, or the reverse (the change in precision). A class can have more broken than fixed pages and still improve in F1 when it loses many false alarms.

| Page type | Support | chapter-v1.6.0 P | R | F1 | | chapter-v1.7.0 P | R | F1 | | Fixed | Broken | Alarms removed | Alarms added |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| advertisement | 106 | 19.9 | 70.8 | **31.1** | │ | 75.0 | 53.8 | **62.6** | │ | 17 | 35 | 286 | 3 |
| appendix | 3 | 0.0 | 0.0 | **0.0** | │ | 0.0 | 0.0 | **0.0** | │ | 0 | 0 | 0 | 0 |
| backCover | 77 | 37.4 | 71.4 | **49.1** | │ | 68.6 | 62.3 | **65.3** | │ | 8 | 15 | 88 | 18 |
| backEndPaper | 35 | 25.0 | 28.6 | **26.7** | │ | 34.5 | 28.6 | **31.2** | │ | 8 | 8 | 27 | 16 |
| backEndSheet | 73 | 16.9 | 39.7 | **23.7** | │ | 70.8 | 69.9 | **70.3** | │ | 33 | 11 | 140 | 18 |
| blank | 390 | 74.9 | 54.4 | **63.0** | │ | 78.0 | 74.6 | **76.3** | │ | 136 | 57 | 53 | 64 |
| calibrationTable | 55 | 100.0 | 90.9 | **95.2** | │ | 42.4 | 96.4 | **58.9** | │ | 4 | 1 | 0 | 72 |
| cover | 22 | 100.0 | 90.9 | **95.2** | │ | 0.0 | 0.0 | **0.0** | │ | 0 | 20 | 0 | 0 |
| dedication | 5 | 100.0 | 40.0 | **57.1** | │ | 40.0 | 80.0 | **53.3** | │ | 3 | 1 | 0 | 6 |
| edge | 60 | 100.0 | 21.7 | **35.6** | │ | 79.6 | 71.7 | **75.4** | │ | 33 | 3 | 0 | 11 |
| flyleaf | 4 | 0.0 | 0.0 | **0.0** | │ | 0.0 | 0.0 | **0.0** | │ | 0 | 0 | 106 | 2 |
| frontCover | 76 | 47.0 | 81.6 | **59.6** | │ | 78.9 | 78.9 | **78.9** | │ | 7 | 9 | 65 | 11 |
| frontEndPaper | 35 | 34.8 | 22.9 | **27.6** | │ | 53.3 | 22.9 | **32.0** | │ | 7 | 7 | 15 | 7 |
| frontEndSheet | 73 | 31.1 | 19.2 | **23.7** | │ | 83.1 | 67.1 | **74.2** | │ | 39 | 4 | 30 | 9 |
| frontispiece | 4 | 40.0 | 50.0 | **44.4** | │ | 7.0 | 75.0 | **12.8** | │ | 1 | 0 | 2 | 39 |
| illustration | 172 | 85.5 | 79.1 | **82.2** | │ | 93.5 | 83.1 | **88.0** | │ | 17 | 10 | 21 | 8 |
| imprimatur | 1 | 0.0 | 0.0 | **0.0** | │ | 8.3 | 100.0 | **15.4** | │ | 1 | 0 | 0 | 11 |
| index | 45 | 35.0 | 62.2 | **44.8** | │ | 56.7 | 75.6 | **64.8** | │ | 12 | 6 | 31 | 5 |
| jacket | 2 | 33.3 | 100.0 | **50.0** | │ | 40.0 | 100.0 | **57.1** | │ | 0 | 0 | 4 | 3 |
| normalPage | 3890 | 97.7 | 81.4 | **88.8** | │ | 96.8 | 96.1 | **96.4** | │ | 580 | 11 | 41 | 91 |
| preface | 8 | 37.5 | 75.0 | **50.0** | │ | 0.0 | 0.0 | **0.0** | │ | 0 | 6 | 9 | 1 |
| sheetMusic | 479 | 96.8 | 100.0 | **98.4** | │ | 92.1 | 100.0 | **95.9** | │ | 0 | 0 | 10 | 35 |
| spine | 23 | 100.0 | 4.3 | **8.3** | │ | 91.7 | 47.8 | **62.9** | │ | 11 | 1 | 0 | 1 |
| table | 70 | 28.2 | 70.0 | **40.2** | │ | 78.6 | 78.6 | **78.6** | │ | 10 | 4 | 110 | 0 |
| tableOfContents | 42 | 45.5 | 47.6 | **46.5** | │ | 78.8 | 61.9 | **69.3** | │ | 10 | 4 | 22 | 5 |
| titlePage | 283 | 79.7 | 84.5 | **82.0** | │ | 89.6 | 88.7 | **89.2** | │ | 24 | 12 | 48 | 16 |

#### Most frequent page-type errors (reference → predicted)

- **chapter-v1.6.0:** normalPage → advertisement (293), normalPage → table (124), blank → backEndSheet (83), normalPage → index (48), normalPage → titlePage (47), normalPage → listOfIllustrations (45), normalPage → flyleaf (44), blank → flyleaf (31), titlePage → frontCover (22), backEndSheet → blank (20)
- **chapter-v1.7.0:** advertisement → normalPage (37), blank → calibrationTable (28), illustration → normalPage (25), normalPage → titlePage (24), normalPage → blank (22), blank → normalPage (20), normalPage → index (19), backEndPaper → blank (18), normalPage → advertisement (18), blank → sheetMusic (18)

### Side, all pages together (108 pages with a DB placement)

chapter-v1.6.0 has no side prediction. Precision is computed over the pages that have a reference placement only.

| Side | Support | chapter-v1.7.0 P | R | F1 |
|---|---:|---:|---:|---:|
| left | 50 | 78.0 | 78.0 | **78.0** |
| right | 50 | 77.6 | 76.0 | **76.8** |
| singlePage | 8 | 66.7 | 75.0 | **70.6** |

Overall side accuracy of chapter-v1.7.0: 76.9. Confusion (reference → predicted): left → left: 39, left → right: 10, left → singlePage: 1, right → left: 10, right → right: 38, right → singlePage: 2, singlePage → left: 1, singlePage → right: 1, singlePage → singlePage: 6

### Document starts, by kind of batch

The long `boundaries/*` batches hold many short documents next to each other: `issues` (periodical issues of one volume), `books` (unrelated short books) and `mixed`. The others are the regular groups: `issues` and `multipart` batches of two or three adjacent documents, `monographs` of unrelated standalone books, `single` of one document (no true starts to find, so every predicted start there is a false alarm).

| Batch kind | Batches | Pages scored | True starts | chapter-v1.6.0: found | predicted | P | R | F1 | | chapter-v1.7.0: found | predicted | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| all | 59 | 9110 | 245 | 131 | 267 | 49.1 | 53.5 | **51.2** | │ | 222 | 233 | 95.3 | 90.6 | **92.9** |
| boundaries/books | 3 | 1194 | 48 | 31 | 65 | 47.7 | 64.6 | **54.9** | │ | 39 | 44 | 88.6 | 81.2 | **84.8** |
| boundaries/issues | 3 | 699 | 71 | 49 | 52 | 94.2 | 69.0 | **79.7** | │ | 71 | 71 | 100.0 | 100.0 | **100.0** |
| boundaries/mixed | 3 | 1151 | 90 | 33 | 49 | 67.3 | 36.7 | **47.5** | │ | 86 | 87 | 98.9 | 95.6 | **97.2** |
| issues | 10 | 340 | 10 | 8 | 8 | 100.0 | 80.0 | **88.9** | │ | 10 | 10 | 100.0 | 100.0 | **100.0** |
| monographs | 10 | 2875 | 16 | 7 | 43 | 16.3 | 43.8 | **23.7** | │ | 10 | 11 | 90.9 | 62.5 | **74.1** |
| multipart | 10 | 1809 | 10 | 3 | 25 | 12.0 | 30.0 | **17.1** | │ | 6 | 10 | 60.0 | 60.0 | **60.0** |
| single | 20 | 1042 | 0 | 0 | 25 | 0.0 | 0.0 | **0.0** | │ | 0 | 0 | 0.0 | 0.0 | **0.0** |

## Consistency set (fit documents: not held-out)

5341 pages in 50 batches; 278 pages have no reference page type and are not scored for it.

### Page type, all pages together

| | Pages | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| chapter-v1.6.0 | 5063 | 89.2 | 92.4 | 35.7 |
| chapter-v1.7.0 | 5063 | 94.1 | 95.6 | 52.2 |

chapter-v1.7.0 fixes 337 pages and breaks 89 (paired McNemar p = 4.9e-35).
On the 6 curated pages alone: chapter-v1.6.0 accuracy 66.7, chapter-v1.7.0 100.0 (fixed 2, broken 0).

#### Per page type

Precision (P), recall (R) and F1 per engine. Two views of the change, which together explain P and R: *fixed*/*broken* count the pages **of that reference class** that chapter-v1.7.0 gets right or wrong where chapter-v1.6.0 did not or did (the change in recall); *alarms removed*/*added* count the pages **of other classes** that chapter-v1.6.0 wrongly labelled as this type and chapter-v1.7.0 no longer does, or the reverse (the change in precision). A class can have more broken than fixed pages and still improve in F1 when it loses many false alarms.

| Page type | Support | chapter-v1.6.0 P | R | F1 | | chapter-v1.7.0 P | R | F1 | | Fixed | Broken | Alarms removed | Alarms added |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| advertisement | 38 | 20.4 | 89.5 | **33.2** | │ | 77.8 | 18.4 | **29.8** | │ | 0 | 27 | 132 | 1 |
| backCover | 14 | 31.0 | 64.3 | **41.9** | │ | 68.4 | 92.9 | **78.8** | │ | 4 | 0 | 17 | 3 |
| backEndSheet | 14 | 27.3 | 42.9 | **33.3** | │ | 76.5 | 92.9 | **83.9** | │ | 7 | 0 | 16 | 4 |
| blank | 60 | 50.0 | 56.7 | **53.1** | │ | 61.6 | 88.3 | **72.6** | │ | 21 | 2 | 15 | 14 |
| frontCover | 15 | 44.4 | 80.0 | **57.1** | │ | 47.8 | 73.3 | **57.9** | │ | 2 | 3 | 12 | 9 |
| frontEndSheet | 16 | 0.0 | 0.0 | **0.0** | │ | 63.2 | 75.0 | **68.6** | │ | 12 | 0 | 2 | 7 |
| frontJacket | 1 | 0.0 | 0.0 | **0.0** | │ | 0.0 | 0.0 | **0.0** | │ | 0 | 0 | 7 | 1 |
| index | 3 | 4.8 | 100.0 | **9.2** | │ | 2.8 | 100.0 | **5.4** | │ | 0 | 0 | 12 | 59 |
| jacket | 1 | 0.0 | 0.0 | **0.0** | │ | 16.7 | 100.0 | **28.6** | │ | 1 | 0 | 1 | 4 |
| map | 1 | 14.3 | 100.0 | **25.0** | │ | 0.0 | 0.0 | **0.0** | │ | 0 | 1 | 3 | 0 |
| normalPage | 4793 | 99.7 | 90.4 | **94.8** | │ | 99.4 | 95.1 | **97.2** | │ | 278 | 55 | 11 | 24 |
| table | 9 | 17.8 | 88.9 | **29.6** | │ | 30.8 | 88.9 | **45.7** | │ | 0 | 0 | 26 | 7 |
| tableOfContents | 16 | 42.9 | 56.2 | **48.6** | │ | 73.7 | 87.5 | **80.0** | │ | 5 | 0 | 12 | 5 |
| titlePage | 82 | 65.1 | 84.1 | **73.4** | │ | 74.3 | 91.5 | **82.0** | │ | 7 | 1 | 29 | 18 |

#### Most frequent page-type errors (reference → predicted)

- **chapter-v1.6.0:** normalPage → advertisement (131), normalPage → flyleaf (68), normalPage → index (58), normalPage → table (37), normalPage → titlePage (30), normalPage → listOfIllustrations (30), normalPage → blank (29), normalPage → backCover (12), normalPage → illustration (11), normalPage → tableOfContents (10)
- **chapter-v1.7.0:** normalPage → index (106), normalPage → blank (29), advertisement → normalPage (23), normalPage → titlePage (23), normalPage → table (18), normalPage → errata (9), normalPage → illustration (6), normalPage → frontispiece (6), normalPage → impressum (6), normalPage → frontCover (4)

### Side, all pages together (246 pages with a DB placement)

chapter-v1.6.0 has no side prediction. Precision is computed over the pages that have a reference placement only.

| Side | Support | chapter-v1.7.0 P | R | F1 |
|---|---:|---:|---:|---:|
| left | 121 | 100.0 | 98.3 | **99.2** |
| right | 121 | 99.2 | 99.2 | **99.2** |
| singlePage | 4 | 50.0 | 75.0 | **60.0** |

Overall side accuracy of chapter-v1.7.0: 98.4. Confusion (reference → predicted): left → left: 119, left → singlePage: 2, right → right: 120, right → singlePage: 1, singlePage → right: 1, singlePage → singlePage: 3

### Document starts, by kind of batch

The long `boundaries/*` batches hold many short documents next to each other: `issues` (periodical issues of one volume), `books` (unrelated short books) and `mixed`. The others are the regular groups: `issues` and `multipart` batches of two or three adjacent documents, `monographs` of unrelated standalone books, `single` of one document (no true starts to find, so every predicted start there is a false alarm).

| Batch kind | Batches | Pages scored | True starts | chapter-v1.6.0: found | predicted | P | R | F1 | | chapter-v1.7.0: found | predicted | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| all | 50 | 5291 | 40 | 21 | 62 | 33.9 | 52.5 | **41.2** | │ | 28 | 31 | 90.3 | 70.0 | **78.9** |
| issues | 10 | 296 | 15 | 10 | 14 | 71.4 | 66.7 | **69.0** | │ | 13 | 13 | 100.0 | 86.7 | **92.9** |
| monographs | 10 | 2646 | 15 | 9 | 30 | 30.0 | 60.0 | **40.0** | │ | 10 | 11 | 90.9 | 66.7 | **76.9** |
| multipart | 10 | 2082 | 10 | 2 | 18 | 11.1 | 20.0 | **14.3** | │ | 5 | 6 | 83.3 | 50.0 | **62.5** |
| single | 20 | 267 | 0 | 0 | 0 | 0.0 | 0.0 | **0.0** | │ | 0 | 1 | 0.0 | 0.0 | **0.0** |


# The page-number engine against the DB page numbers

The page-number stage of MetaKat, scored against the DB page numbers, the cataloguing labels, with the DB label as the truth. A page is right only when the predicted label is identical to the DB label (`12`, `[3]` and `[12a]` are different labels). *Other convention* is a label with the right number written in another convention that the standard also allows or that the DB uses (`12` for `[12]`, `[12a]` for `12v`, `12r` for `12`); it is not counted as right, and shows how much of the error is the convention, not the number. *Missed* is a page where the DB has a number and the engine output none, *wrong* a page where it output a different number. The numbers come from the same MetaKat runs as the page types. The engine versions differ in the page-number logic that lives in the code, not in the model, so the **old code** column is a run with the code from before the page-number changes (the filling of gaps, bracketed and suffixed labels from the document context) and the **new code** columns are the current code.

The pages are split into the situations of the NDK pagination rules by the DB label: *normal numbers* (a plain numeral, printed or filled into a gap), *bracketed numbers at the beginning* (before the first plain number of the document), *bracketed numbers in the middle* (after it) and *bracketed numbers with a letter suffix* (inserted leaves, unnumbered leaves before page 1). Other labels (roman numerals, signatures, recto/verso) are only in the total. Percentages.

## Evaluation (dev documents)

| Situation | Pages | chapter-v1.6.0: right | other convention | missed | wrong | | chapter-v1.7.0: right | other convention | missed | wrong |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| All pages with a DB page number | 9153 | 37.7 | 0.6 | 57.1 | 4.6 | │ | 44.9 | 1.9 | 47.4 | 5.8 |
| Normal numbers (`12`) | 6125 | 55.9 | 0.0 | 40.2 | 3.8 | │ | 65.0 | 0.6 | 31.6 | 2.7 |
| Bracketed numbers at the beginning (`[3]`, before the first plain number) | 592 | 0.0 | 0.2 | 96.3 | 3.5 | │ | 6.4 | 0.2 | 89.4 | 4.1 |
| Bracketed numbers in the middle (`[3]`, after the first plain number) | 594 | 0.0 | 0.0 | 85.0 | 15.0 | │ | 9.3 | 3.2 | 65.2 | 22.4 |
| Bracketed numbers with a letter suffix (`[12a]`, `[1b]`) | 712 | 0.0 | 0.0 | 95.4 | 4.6 | │ | 2.0 | 0.6 | 92.3 | 5.2 |

16 pages have no DB page number and are not scored. Pages where the variant predicts a number anyway: chapter-v1.6.0 6, chapter-v1.7.0 6.

## Consistency (fit documents)

| Situation | Pages | chapter-v1.6.0: right | other convention | missed | wrong | | chapter-v1.7.0: right | other convention | missed | wrong |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| All pages with a DB page number | 5305 | 73.6 | 0.0 | 23.1 | 3.3 | │ | 78.0 | 1.3 | 17.7 | 3.0 |
| Normal numbers (`12`) | 4818 | 81.1 | 0.0 | 16.0 | 2.9 | │ | 84.8 | 0.3 | 12.5 | 2.3 |
| Bracketed numbers at the beginning (`[3]`, before the first plain number) | 194 | 0.0 | 0.5 | 94.8 | 4.6 | │ | 13.4 | 1.5 | 79.4 | 5.7 |
| Bracketed numbers in the middle (`[3]`, after the first plain number) | 160 | 0.0 | 0.0 | 90.0 | 10.0 | │ | 6.9 | 26.9 | 55.6 | 10.6 |
| Bracketed numbers with a letter suffix (`[12a]`, `[1b]`) | 116 | 0.0 | 0.9 | 95.7 | 3.4 | │ | 10.3 | 7.8 | 69.0 | 12.9 |

36 pages have no DB page number and are not scored. Pages where the variant predicts a number anyway: chapter-v1.6.0 7, chapter-v1.7.0 7.


# The context system on its own: the labeled dev set

The page-type model that MetaKat now runs (the exported W512 fine-tuning checkpoint, step 15927, `finetune_best.pt`, SHA-256 `e52a49a453ce844e…`), evaluated by the training pipeline on its native features: the DINOv2, mmBERT and page-statistics vectors of the training store, not the MetaKat extraction. 193 dev streams (431,396 pages) are each scanned whole with 512-page windows, stride 256 and centre-weighted merging, as in deployment; the scan endpoints are inputs in every case. The deployment export gives the same decisions as this checkpoint (see `metakat-boundary5-inference-parity.md`).

- **Page type:** the curated (annotated) pages only, so the reference is a human label.
- **Side:** the DB placement (`left`, `right`, `single`), on the pages that have one.
- **Document starts:** the true document boundaries of the streams, every page after the first of a stream; the decision threshold is 0.5. AP is average precision over the start probability.
- **Optional inputs:** the *page numbers* (pagination features; absent means all zero) and the *document boundaries* (the true starts and ends of every document, given as input). The model was trained to work without either, with the boundaries supplied for half of its training scans.
- The dev split also chose this checkpoint (curated page-type F1 without extra input), so it is slightly optimistic. The test split has no pagination, so it cannot be used for all four cases.
- Percentages; confusion matrices have the reference in the rows and the prediction in the columns, with the correct cells in bold.

## Summary

### Page type

| Input | Pages | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| No extra input | 3758 | 91.4 | **91.3** | **71.3** |
| Page numbers | 3758 | 91.4 | **91.3** | **71.4** |
| Document boundaries | 3758 | 92.2 | **92.1** | **72.7** |
| Page numbers and boundaries | 3758 | 92.3 | **92.3** | **72.7** |

### Side

| Input | Pages | Accuracy | Weighted F1 |
|---|---:|---:|---:|
| No extra input | 111193 | 88.7 | **88.7** |
| Page numbers | 111193 | 94.9 | **94.9** |
| Document boundaries | 111193 | 88.8 | **88.8** |
| Page numbers and boundaries | 111193 | 95.0 | **95.0** |

### Document starts

| Input | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| No extra input | 92.4 | 90.3 | **91.3** | 96.1 |
| Page numbers | 92.6 | 90.3 | **91.5** | 96.3 |
| Document boundaries | 99.0 | 99.8 | **99.4** | 100.0 |
| Page numbers and boundaries | 98.8 | 99.9 | **99.3** | 100.0 |

Documents recovered exactly (start and end both right), of 1979: no extra input 80.0%, page numbers 79.9%, document boundaries 98.9%, page numbers and boundaries 98.8%.

## No extra input

Page numbers absent, document boundaries absent.

### Page type

3758 curated pages: accuracy 91.4, weighted F1 91.3, macro F1 71.3.

| Page type | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Advertisement | 17 | 86.7 | 76.5 | **81.2** |
| Appendix | 2 | 0.0 | 0.0 | **0.0** |
| BackCover | 276 | 94.5 | 93.8 | **94.2** |
| BackEndPaper | 6 | 57.1 | 66.7 | **61.5** |
| BackEndSheet | 276 | 97.4 | 95.7 | **96.5** |
| Bibliography | 5 | 100.0 | 40.0 | **57.1** |
| Blank | 154 | 86.2 | 89.6 | **87.9** |
| CalibrationTable | 495 | 99.8 | 99.4 | **99.6** |
| Colophon | 2 | 0.0 | 0.0 | **0.0** |
| Cover | 12 | 85.7 | 50.0 | **63.2** |
| Errata | 6 | 66.7 | 66.7 | **66.7** |
| FlyLeaf | 178 | 83.0 | 84.8 | **83.9** |
| FragmentsOfBookbinding | 1 | 0.0 | 0.0 | **0.0** |
| FrontCover | 266 | 86.3 | 94.4 | **90.1** |
| FrontEndPaper | 1 | 0.0 | 0.0 | **0.0** |
| FrontEndSheet | 44 | 83.3 | 90.9 | **87.0** |
| FrontJacket | 204 | 94.5 | 92.6 | **93.6** |
| Frontispiece | 11 | 88.9 | 72.7 | **80.0** |
| Illustration | 42 | 78.1 | 59.5 | **67.6** |
| Impressum | 2 | 22.2 | 100.0 | **36.4** |
| Imprimatur | 3 | 100.0 | 100.0 | **100.0** |
| Index | 136 | 97.1 | 97.1 | **97.1** |
| Jacket | 64 | 93.7 | 92.2 | **92.9** |
| ListOfIllustrations | 84 | 80.9 | 65.5 | **72.4** |
| ListOfMaps | 12 | 78.6 | 91.7 | **84.6** |
| ListOfTables | 4 | 50.0 | 75.0 | **60.0** |
| Map | 195 | 92.0 | 93.8 | **92.9** |
| NormalPage | 369 | 78.8 | 82.4 | **80.5** |
| Preface | 1 | 100.0 | 100.0 | **100.0** |
| SheetMusic | 133 | 97.8 | 99.2 | **98.5** |
| Spine | 193 | 100.0 | 99.5 | **99.7** |
| Table | 112 | 89.2 | 81.2 | **85.0** |
| TableOfContents | 183 | 93.4 | 92.3 | **92.9** |
| TitlePage | 269 | 95.1 | 93.3 | **94.2** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Advertisement | **13** | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 2 Appendix | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| 3 BackCover | · | · | **259** | · | 3 | · | 3 | · | · | · | · | · | 2 | · | 1 | · | 2 | 2 | · | · | · | · | · | 2 | · | · | · | 1 | · | · | · | · | · | 1 | · |
| 4 BackEndPaper | · | · | 1 | **4** | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 5 BackEndSheet | · | · | 2 | 1 | **264** | · | 4 | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | 1 | · |
| 6 Bibliography | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 7 Blank | · | · | 2 | · | · | · | **138** | · | · | · | · | · | 4 | · | · | 1 | 3 | · | · | · | 1 | · | · | · | · | · | · | · | 5 | · | · | · | · | · | · |
| 8 CalibrationTable | · | · | · | · | · | · | · | **492** | · | · | · | · | · | · | 1 | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · |
| 9 Colophon | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 10 Cover | · | · | 1 | · | · | · | · | · | · | **6** | · | · | · | · | 4 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 11 Dedication | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 12 Errata | · | · | · | · | · | · | · | · | · | · | · | **4** | 2 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 13 FlyLeaf | · | · | 3 | · | 1 | · | 6 | · | · | · | · | · | **151** | · | · | · | 1 | 1 | · | 1 | · | · | · | 1 | 1 | · | · | 4 | 5 | · | 1 | · | 2 | · | · |
| 14 FragmentsOfBookbinding | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 15 FrontCover | · | · | 1 | · | · | · | · | · | · | · | · | · | 2 | · | **251** | · | · | 5 | · | · | · | · | · | 1 | · | · | · | · | 1 | · | · | · | · | · | 5 |
| 16 FrontEndPaper | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 17 FrontEndSheet | · | · | · | · | · | · | 3 | · | · | · | · | · | · | · | 1 | · | **40** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 18 FrontJacket | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | 13 | · | · | **189** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 19 Frontispiece | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **8** | 1 | 1 | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 20 Illustration | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | **25** | · | · | · | · | · | · | · | 4 | 12 | · | · | · | · | · | · |
| 21 Impressum | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 22 Imprimatur | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **3** | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 23 Index | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **132** | · | 1 | · | · | · | 3 | · | · | · | · | · | · |
| 24 Jacket | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | **59** | · | · | · | · | 1 | · | · | · | · | · | 1 |
| 25 ListOfIllustrations | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | 2 | · | **55** | 3 | 2 | · | 11 | · | · | · | 1 | 9 | · |
| 26 ListOfMaps | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **11** | 1 | · | · | · | · | · | · | · | · |
| 27 ListOfTables | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **3** | · | · | · | · | · | · | · | · |
| 28 Map | · | · | · | · | 2 | · | 1 | · | · | · | · | · | 4 | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | **183** | 4 | · | · | · | · | · | · |
| 29 NormalPage | 2 | · | 2 | 2 | · | · | 2 | · | · | 1 | 1 | 1 | 12 | · | 1 | · | · | · | · | 4 | 4 | · | 2 | · | 9 | · | · | 7 | **304** | · | 1 | · | 8 | 1 | 5 |
| 30 Preface | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **1** | · | · | · | · | · |
| 31 SheetMusic | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **132** | · | · | · | · |
| 32 Spine | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | **192** | · | · | · |
| 33 Table | · | · | · | · | · | · | 1 | · | · | · | · | · | 3 | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | 16 | · | · | · | **91** | · | · |
| 34 TableOfContents | · | · | 1 | · | 1 | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | 1 | · | · | · | 6 | · | · | · | · | **169** | 2 |
| 35 TitlePage | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 12 | · | · | 1 | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · | · | · | **251** |

Most frequent mistakes (reference → predicted): Table → NormalPage (16), FrontJacket → FrontCover (13), TitlePage → FrontCover (12), NormalPage → FlyLeaf (12), Illustration → NormalPage (12), ListOfIllustrations → NormalPage (11), NormalPage → ListOfIllustrations (9), ListOfIllustrations → TableOfContents (9), NormalPage → Table (8), NormalPage → Map (7), TableOfContents → NormalPage (6), FlyLeaf → Blank (6)

### Side

111193 pages with a DB placement: accuracy 88.7, weighted F1 88.7.

| Side | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| left | 54272 | 88.7 | 88.9 | **88.8** |
| right | 54231 | 88.7 | 89.0 | **88.9** |
| single | 2690 | 88.4 | 80.1 | **84.0** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 |
|---|---:|---:|---:|
| 1 left | **48226** | 5909 | 137 |
| 2 right | 5813 | **48271** | 147 |
| 3 single | 305 | 230 | **2155** |

Most frequent mistakes (reference → predicted): left → right (5909), right → left (5813), single → left (305), single → right (230), right → single (147), left → single (137)

### Document starts

431,203 page transitions, 1786 of them true document starts: precision 92.4, recall 90.3, F1 91.3, AP 96.1, 0.31 false splits per 1,000 non-starts. Documents recovered exactly: 1583 of 1979 (80.0%), with 1938 predicted.

| Document start | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| not a start | 429417 | 100.0 | 100.0 | **100.0** |
| start | 1786 | 92.4 | 90.3 | **91.3** |

Confusion matrix:

| Reference \ predicted | 1 | 2 |
|---|---:|---:|
| 1 not a start | **429284** | 133 |
| 2 start | 174 | **1612** |

Most frequent mistakes (reference → predicted): start → not a start (174), not a start → start (133)

Recall of the true starts by what the neighbouring documents are:

| Boundary between | True starts | Found | Recall |
|---|---:|---:|---:|
| unrelated documents | 1777 | 1608 | 90.5 |
| same volume (adjacent issues or parts) | 9 | 4 | 44.4 |

## Page numbers

Page numbers supplied, document boundaries absent.

### Page type

3758 curated pages: accuracy 91.4, weighted F1 91.3, macro F1 71.4.

| Page type | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Advertisement | 17 | 86.7 | 76.5 | **81.2** |
| Appendix | 2 | 0.0 | 0.0 | **0.0** |
| BackCover | 276 | 94.9 | 93.8 | **94.4** |
| BackEndPaper | 6 | 57.1 | 66.7 | **61.5** |
| BackEndSheet | 276 | 97.1 | 96.0 | **96.5** |
| Bibliography | 5 | 100.0 | 40.0 | **57.1** |
| Blank | 154 | 86.2 | 89.6 | **87.9** |
| CalibrationTable | 495 | 99.8 | 99.2 | **99.5** |
| Colophon | 2 | 0.0 | 0.0 | **0.0** |
| Cover | 12 | 75.0 | 50.0 | **60.0** |
| Errata | 6 | 66.7 | 66.7 | **66.7** |
| FlyLeaf | 178 | 83.9 | 87.6 | **85.7** |
| FragmentsOfBookbinding | 1 | 0.0 | 0.0 | **0.0** |
| FrontCover | 266 | 86.5 | 93.6 | **89.9** |
| FrontEndPaper | 1 | 0.0 | 0.0 | **0.0** |
| FrontEndSheet | 44 | 85.1 | 90.9 | **87.9** |
| FrontJacket | 204 | 94.5 | 93.1 | **93.8** |
| Frontispiece | 11 | 88.9 | 72.7 | **80.0** |
| Illustration | 42 | 76.7 | 54.8 | **63.9** |
| Impressum | 2 | 20.0 | 100.0 | **33.3** |
| Imprimatur | 3 | 100.0 | 100.0 | **100.0** |
| Index | 136 | 97.1 | 97.1 | **97.1** |
| Jacket | 64 | 93.7 | 92.2 | **92.9** |
| ListOfIllustrations | 84 | 81.7 | 69.0 | **74.8** |
| ListOfMaps | 12 | 84.6 | 91.7 | **88.0** |
| ListOfTables | 4 | 60.0 | 75.0 | **66.7** |
| Map | 195 | 91.5 | 94.4 | **92.9** |
| NormalPage | 369 | 78.7 | 81.3 | **80.0** |
| Preface | 1 | 100.0 | 100.0 | **100.0** |
| SheetMusic | 133 | 97.1 | 99.2 | **98.1** |
| Spine | 193 | 100.0 | 99.5 | **99.7** |
| Table | 112 | 88.9 | 78.6 | **83.4** |
| TableOfContents | 183 | 93.3 | 91.8 | **92.6** |
| TitlePage | 269 | 94.4 | 93.7 | **94.0** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Advertisement | **13** | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 2 Appendix | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| 3 BackCover | · | · | **259** | · | 3 | · | 3 | · | · | · | · | · | 2 | · | 2 | · | 1 | 2 | · | · | · | · | · | 1 | · | · | · | 1 | 1 | · | · | · | · | 1 | · |
| 4 BackEndPaper | · | · | 1 | **4** | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 5 BackEndSheet | · | · | 2 | 1 | **265** | · | 3 | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | 1 | · |
| 6 Bibliography | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 7 Blank | · | · | 1 | · | · | · | **138** | · | · | 1 | · | · | 4 | · | · | 2 | 3 | · | · | · | 1 | · | · | · | · | · | · | · | 4 | · | · | · | · | · | · |
| 8 CalibrationTable | · | · | · | · | · | · | · | **491** | · | · | · | · | · | · | 1 | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · |
| 9 Colophon | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 10 Cover | · | · | · | · | · | · | · | · | · | **6** | · | · | · | · | 5 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 11 Dedication | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 12 Errata | · | · | · | · | · | · | · | · | · | · | · | **4** | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 13 FlyLeaf | · | · | 3 | · | · | · | 6 | · | · | · | · | · | **156** | · | · | · | 1 | · | · | 1 | · | · | · | 1 | 1 | · | · | 4 | 2 | · | 1 | · | 2 | · | · |
| 14 FragmentsOfBookbinding | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 15 FrontCover | · | · | 2 | · | · | · | · | · | · | · | · | · | 1 | · | **249** | · | · | 6 | · | · | · | · | · | 2 | · | · | · | · | 1 | · | · | · | · | · | 5 |
| 16 FrontEndPaper | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 17 FrontEndSheet | · | · | · | · | · | · | 3 | · | · | · | · | · | · | · | 1 | · | **40** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 18 FrontJacket | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | 12 | · | · | **190** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 19 Frontispiece | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **8** | 1 | 1 | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 20 Illustration | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | **23** | · | · | · | · | · | · | · | 5 | 13 | · | · | · | · | · | · |
| 21 Impressum | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 22 Imprimatur | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **3** | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 23 Index | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **132** | · | 1 | · | · | · | 3 | · | · | · | · | · | · |
| 24 Jacket | · | · | 1 | · | 1 | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | **59** | · | · | · | · | · | · | · | · | · | · | 1 |
| 25 ListOfIllustrations | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | 2 | · | **58** | 2 | 1 | · | 9 | · | · | · | 1 | 9 | · |
| 26 ListOfMaps | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **11** | 1 | · | · | · | · | · | · | · | · |
| 27 ListOfTables | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **3** | · | · | · | · | · | · | · | · |
| 28 Map | · | · | · | · | 2 | · | 1 | · | · | · | · | · | 4 | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | **184** | 3 | · | · | · | · | · | · |
| 29 NormalPage | 2 | · | 2 | 2 | · | · | 3 | · | · | 1 | 1 | 1 | 13 | · | 1 | · | · | · | · | 5 | 4 | · | 2 | · | 9 | · | · | 7 | **300** | · | 1 | · | 8 | 1 | 6 |
| 30 Preface | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **1** | · | · | · | · | · |
| 31 SheetMusic | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **132** | · | · | · | · |
| 32 Spine | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | **192** | · | · | · |
| 33 Table | · | · | · | · | · | · | 1 | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 20 | · | · | · | **88** | · | · |
| 34 TableOfContents | · | · | 1 | · | 1 | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | 1 | · | · | · | 1 | · | · | · | 6 | · | · | · | · | **168** | 3 |
| 35 TitlePage | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 11 | · | · | 1 | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · | · | · | **252** |

Most frequent mistakes (reference → predicted): Table → NormalPage (20), NormalPage → FlyLeaf (13), Illustration → NormalPage (13), FrontJacket → FrontCover (12), TitlePage → FrontCover (11), NormalPage → ListOfIllustrations (9), ListOfIllustrations → TableOfContents (9), ListOfIllustrations → NormalPage (9), NormalPage → Table (8), NormalPage → Map (7), TableOfContents → NormalPage (6), NormalPage → TitlePage (6)

### Side

111193 pages with a DB placement: accuracy 94.9, weighted F1 94.9.

| Side | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| left | 54272 | 95.0 | 95.2 | **95.1** |
| right | 54231 | 95.0 | 95.3 | **95.2** |
| single | 2690 | 91.6 | 82.6 | **86.9** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 |
|---|---:|---:|---:|
| 1 left | **51679** | 2491 | 102 |
| 2 right | 2455 | **51675** | 101 |
| 3 single | 265 | 204 | **2221** |

Most frequent mistakes (reference → predicted): left → right (2491), right → left (2455), single → left (265), single → right (204), left → single (102), right → single (101)

### Document starts

431,203 page transitions, 1786 of them true document starts: precision 92.6, recall 90.3, F1 91.5, AP 96.3, 0.30 false splits per 1,000 non-starts. Documents recovered exactly: 1581 of 1979 (79.9%), with 1934 predicted.

| Document start | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| not a start | 429417 | 100.0 | 100.0 | **100.0** |
| start | 1786 | 92.6 | 90.3 | **91.5** |

Confusion matrix:

| Reference \ predicted | 1 | 2 |
|---|---:|---:|
| 1 not a start | **429289** | 128 |
| 2 start | 173 | **1613** |

Most frequent mistakes (reference → predicted): start → not a start (173), not a start → start (128)

Recall of the true starts by what the neighbouring documents are:

| Boundary between | True starts | Found | Recall |
|---|---:|---:|---:|
| unrelated documents | 1777 | 1608 | 90.5 |
| same volume (adjacent issues or parts) | 9 | 5 | 55.6 |

## Document boundaries

Page numbers absent, document boundaries supplied.

### Page type

3758 curated pages: accuracy 92.2, weighted F1 92.1, macro F1 72.7.

| Page type | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Advertisement | 17 | 86.7 | 76.5 | **81.2** |
| Appendix | 2 | 0.0 | 0.0 | **0.0** |
| BackCover | 276 | 97.1 | 96.4 | **96.7** |
| BackEndPaper | 6 | 66.7 | 66.7 | **66.7** |
| BackEndSheet | 276 | 97.5 | 98.2 | **97.8** |
| Bibliography | 5 | 100.0 | 40.0 | **57.1** |
| Blank | 154 | 89.7 | 90.3 | **90.0** |
| CalibrationTable | 495 | 99.8 | 99.8 | **99.8** |
| Colophon | 2 | 0.0 | 0.0 | **0.0** |
| Cover | 12 | 85.7 | 50.0 | **63.2** |
| Errata | 6 | 71.4 | 83.3 | **76.9** |
| FlyLeaf | 178 | 83.6 | 88.8 | **86.1** |
| FragmentsOfBookbinding | 1 | 0.0 | 0.0 | **0.0** |
| FrontCover | 266 | 87.9 | 95.9 | **91.7** |
| FrontEndPaper | 1 | 0.0 | 0.0 | **0.0** |
| FrontEndSheet | 44 | 97.6 | 93.2 | **95.3** |
| FrontJacket | 204 | 95.1 | 94.6 | **94.8** |
| Frontispiece | 11 | 100.0 | 72.7 | **84.2** |
| Illustration | 42 | 75.8 | 59.5 | **66.7** |
| Impressum | 2 | 22.2 | 100.0 | **36.4** |
| Imprimatur | 3 | 100.0 | 100.0 | **100.0** |
| Index | 136 | 96.4 | 97.1 | **96.7** |
| Jacket | 64 | 96.8 | 93.8 | **95.2** |
| ListOfIllustrations | 84 | 80.6 | 64.3 | **71.5** |
| ListOfMaps | 12 | 78.6 | 91.7 | **84.6** |
| ListOfTables | 4 | 60.0 | 75.0 | **66.7** |
| Map | 195 | 91.0 | 92.8 | **91.9** |
| NormalPage | 369 | 78.8 | 81.6 | **80.2** |
| Preface | 1 | 100.0 | 100.0 | **100.0** |
| SheetMusic | 133 | 99.3 | 100.0 | **99.6** |
| Spine | 193 | 99.5 | 99.5 | **99.5** |
| Table | 112 | 89.2 | 81.2 | **85.0** |
| TableOfContents | 183 | 93.4 | 92.9 | **93.2** |
| TitlePage | 269 | 95.8 | 92.9 | **94.3** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Advertisement | **13** | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 2 Appendix | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| 3 BackCover | · | · | **266** | · | 1 | · | 2 | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | 1 | · | · | · | · | · | 1 | 1 |
| 4 BackEndPaper | · | · | · | **4** | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · |
| 5 BackEndSheet | · | · | 1 | · | **271** | · | 2 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · |
| 6 Bibliography | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 7 Blank | · | · | 1 | · | · | · | **139** | · | · | · | · | · | 5 | · | · | 2 | 1 | · | · | · | 1 | · | · | · | · | · | · | · | 5 | · | · | · | · | · | · |
| 8 CalibrationTable | · | · | · | · | · | · | · | **494** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 9 Colophon | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 10 Cover | · | · | · | · | · | · | · | · | · | **6** | · | · | · | · | 5 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 11 Dedication | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 12 Errata | · | · | · | · | · | · | · | · | · | · | · | **5** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 13 FlyLeaf | · | · | 2 | · | 1 | · | 4 | · | · | · | · | · | **158** | · | · | · | · | · | · | 1 | · | · | · | 1 | 1 | · | · | 4 | 3 | · | · | 1 | 2 | · | · |
| 14 FragmentsOfBookbinding | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 15 FrontCover | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **255** | · | · | 7 | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | 3 |
| 16 FrontEndPaper | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 17 FrontEndSheet | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | 1 | · | **41** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 18 FrontJacket | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 11 | · | · | **193** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 19 Frontispiece | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **8** | 1 | 1 | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 20 Illustration | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | **25** | · | · | · | · | · | · | · | 4 | 12 | · | · | · | · | · | · |
| 21 Impressum | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 22 Imprimatur | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **3** | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 23 Index | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **132** | · | 1 | · | · | · | 3 | · | · | · | · | · | · |
| 24 Jacket | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | **60** | · | · | · | · | 1 | · | · | · | · | · | · |
| 25 ListOfIllustrations | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | **54** | 3 | 1 | · | 11 | · | · | · | 1 | 9 | · |
| 26 ListOfMaps | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **11** | 1 | · | · | · | · | · | · | · | · |
| 27 ListOfTables | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **3** | · | · | · | · | · | · | · | · |
| 28 Map | · | · | · | · | 2 | · | 1 | · | · | · | · | · | 5 | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | **181** | 5 | · | · | · | · | · | · |
| 29 NormalPage | 2 | · | 1 | 2 | · | · | 2 | · | · | 1 | 1 | 1 | 14 | · | 1 | · | · | · | · | 5 | 4 | · | 2 | · | 9 | · | · | 8 | **301** | · | 1 | · | 8 | 1 | 5 |
| 30 Preface | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **1** | · | · | · | · | · |
| 31 SheetMusic | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **133** | · | · | · | · |
| 32 Spine | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | **192** | · | · | · |
| 33 Table | · | · | · | · | · | · | 1 | · | · | · | · | · | 3 | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | 16 | · | · | · | **91** | · | · |
| 34 TableOfContents | · | · | 1 | · | 1 | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | 1 | · | · | · | 6 | · | · | · | · | **170** | 2 |
| 35 TitlePage | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 13 | · | · | 1 | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · | · | · | **250** |

Most frequent mistakes (reference → predicted): Table → NormalPage (16), NormalPage → FlyLeaf (14), TitlePage → FrontCover (13), Illustration → NormalPage (12), ListOfIllustrations → NormalPage (11), FrontJacket → FrontCover (11), NormalPage → ListOfIllustrations (9), ListOfIllustrations → TableOfContents (9), NormalPage → Table (8), NormalPage → Map (8), FrontCover → FrontJacket (7), TableOfContents → NormalPage (6)

### Side

111193 pages with a DB placement: accuracy 88.8, weighted F1 88.8.

| Side | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| left | 54272 | 88.7 | 89.0 | **88.8** |
| right | 54231 | 88.8 | 89.0 | **88.9** |
| single | 2690 | 90.5 | 79.8 | **84.8** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 |
|---|---:|---:|---:|
| 1 left | **48293** | 5875 | 104 |
| 2 right | 5849 | **48260** | 122 |
| 3 single | 303 | 240 | **2147** |

Most frequent mistakes (reference → predicted): left → right (5875), right → left (5849), single → left (303), single → right (240), right → single (122), left → single (104)

### Document starts

431,203 page transitions, 1786 of them true document starts: precision 99.0, recall 99.8, F1 99.4, AP 100.0, 0.04 false splits per 1,000 non-starts. Documents recovered exactly: 1957 of 1979 (98.9%), with 1994 predicted.

| Document start | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| not a start | 429417 | 100.0 | 100.0 | **100.0** |
| start | 1786 | 99.0 | 99.8 | **99.4** |

Confusion matrix:

| Reference \ predicted | 1 | 2 |
|---|---:|---:|
| 1 not a start | **429399** | 18 |
| 2 start | 3 | **1783** |

Most frequent mistakes (reference → predicted): not a start → start (18), start → not a start (3)

Recall of the true starts by what the neighbouring documents are:

| Boundary between | True starts | Found | Recall |
|---|---:|---:|---:|
| unrelated documents | 1777 | 1774 | 99.8 |
| same volume (adjacent issues or parts) | 9 | 9 | 100.0 |

## Page numbers and boundaries

Page numbers supplied, document boundaries supplied.

### Page type

3758 curated pages: accuracy 92.3, weighted F1 92.3, macro F1 72.7.

| Page type | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Advertisement | 17 | 87.5 | 82.4 | **84.8** |
| Appendix | 2 | 0.0 | 0.0 | **0.0** |
| BackCover | 276 | 97.1 | 96.7 | **96.9** |
| BackEndPaper | 6 | 66.7 | 66.7 | **66.7** |
| BackEndSheet | 276 | 97.8 | 98.6 | **98.2** |
| Bibliography | 5 | 100.0 | 40.0 | **57.1** |
| Blank | 154 | 90.3 | 90.3 | **90.3** |
| CalibrationTable | 495 | 99.8 | 99.8 | **99.8** |
| Colophon | 2 | 0.0 | 0.0 | **0.0** |
| Cover | 12 | 85.7 | 50.0 | **63.2** |
| Errata | 6 | 71.4 | 83.3 | **76.9** |
| FlyLeaf | 178 | 84.6 | 89.3 | **86.9** |
| FragmentsOfBookbinding | 1 | 0.0 | 0.0 | **0.0** |
| FrontCover | 266 | 88.5 | 95.9 | **92.1** |
| FrontEndPaper | 1 | 0.0 | 0.0 | **0.0** |
| FrontEndSheet | 44 | 95.3 | 93.2 | **94.3** |
| FrontJacket | 204 | 95.1 | 95.6 | **95.4** |
| Frontispiece | 11 | 100.0 | 72.7 | **84.2** |
| Illustration | 42 | 76.7 | 54.8 | **63.9** |
| Impressum | 2 | 18.2 | 100.0 | **30.8** |
| Imprimatur | 3 | 100.0 | 100.0 | **100.0** |
| Index | 136 | 97.1 | 97.1 | **97.1** |
| Jacket | 64 | 93.8 | 93.8 | **93.8** |
| ListOfIllustrations | 84 | 81.7 | 69.0 | **74.8** |
| ListOfMaps | 12 | 84.6 | 91.7 | **88.0** |
| ListOfTables | 4 | 60.0 | 75.0 | **66.7** |
| Map | 195 | 91.0 | 93.8 | **92.4** |
| NormalPage | 369 | 78.6 | 81.6 | **80.1** |
| Preface | 1 | 100.0 | 100.0 | **100.0** |
| SheetMusic | 133 | 99.2 | 99.2 | **99.2** |
| Spine | 193 | 99.5 | 99.5 | **99.5** |
| Table | 112 | 88.9 | 78.6 | **83.4** |
| TableOfContents | 183 | 93.9 | 92.3 | **93.1** |
| TitlePage | 269 | 96.2 | 92.9 | **94.5** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Advertisement | **14** | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| 2 Appendix | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| 3 BackCover | · | · | **267** | · | 1 | · | 2 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | 1 | · | · | · | · | · | 1 | 1 |
| 4 BackEndPaper | · | · | · | **4** | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · |
| 5 BackEndSheet | · | · | 1 | · | **272** | · | 1 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · |
| 6 Bibliography | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 3 | · | · | · | · | · | · |
| 7 Blank | · | · | 1 | · | · | · | **139** | · | · | · | · | · | 5 | · | · | 2 | 1 | · | · | · | 2 | · | · | · | · | · | · | · | 4 | · | · | · | · | · | · |
| 8 CalibrationTable | · | · | · | · | · | · | · | **494** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 9 Colophon | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 10 Cover | · | · | · | · | · | · | · | · | · | **6** | · | · | · | · | 5 | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 11 Dedication | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 12 Errata | · | · | · | · | · | · | · | · | · | · | · | **5** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 13 FlyLeaf | · | · | 2 | · | · | · | 4 | · | · | · | · | · | **159** | · | · | · | · | · | · | 1 | · | · | · | 1 | 1 | · | · | 4 | 3 | · | · | 1 | 2 | · | · |
| 14 FragmentsOfBookbinding | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 15 FrontCover | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **255** | · | · | 7 | · | · | · | · | · | 1 | · | · | · | · | 1 | · | · | · | · | · | 2 |
| 16 FrontEndPaper | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 17 FrontEndSheet | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | 1 | · | **41** | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 18 FrontJacket | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 9 | · | · | **195** | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 19 Frontispiece | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **8** | 1 | 1 | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · |
| 20 Illustration | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | **23** | · | · | · | · | · | · | · | 5 | 13 | · | · | · | · | · | · |
| 21 Impressum | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **2** | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 22 Imprimatur | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **3** | · | · | · | · | · | · | · | · | · | · | · | · | · |
| 23 Index | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **132** | · | 1 | · | · | · | 3 | · | · | · | · | · | · |
| 24 Jacket | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | · | · | · | **60** | · | · | · | · | 1 | · | · | · | · | · | · |
| 25 ListOfIllustrations | · | · | · | · | 2 | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | 2 | · | **58** | 2 | 1 | · | 9 | · | · | · | 1 | 8 | · |
| 26 ListOfMaps | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **11** | 1 | · | · | · | · | · | · | · | · |
| 27 ListOfTables | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **3** | · | · | · | · | · | · | · | · |
| 28 Map | · | · | · | · | 2 | · | 1 | · | · | · | · | · | 5 | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | **183** | 3 | · | · | · | · | · | · |
| 29 NormalPage | 2 | · | 1 | 2 | · | · | 3 | · | · | 1 | 1 | 1 | 14 | · | 1 | · | · | · | · | 5 | 4 | · | 2 | · | 9 | · | · | 7 | **301** | · | 1 | · | 8 | 1 | 5 |
| 30 Preface | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | **1** | · | · | · | · | · |
| 31 SheetMusic | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | **132** | · | · | · | · |
| 32 Spine | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | **192** | · | · | · |
| 33 Table | · | · | · | · | · | · | 1 | · | · | · | · | · | 3 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 20 | · | · | · | **88** | · | · |
| 34 TableOfContents | · | · | 1 | · | 1 | · | · | · | · | · | · | · | · | · | 2 | · | · | · | · | · | 1 | · | · | · | 1 | · | · | · | 6 | · | · | · | · | **169** | 2 |
| 35 TitlePage | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 13 | · | · | 1 | · | · | · | · | · | · | · | · | · | · | 5 | · | · | · | · | · | **250** |

Most frequent mistakes (reference → predicted): Table → NormalPage (20), NormalPage → FlyLeaf (14), TitlePage → FrontCover (13), Illustration → NormalPage (13), NormalPage → ListOfIllustrations (9), ListOfIllustrations → NormalPage (9), FrontJacket → FrontCover (9), NormalPage → Table (8), ListOfIllustrations → TableOfContents (8), NormalPage → Map (7), FrontCover → FrontJacket (7), TableOfContents → NormalPage (6)

### Side

111193 pages with a DB placement: accuracy 95.0, weighted F1 95.0.

| Side | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| left | 54272 | 95.0 | 95.3 | **95.1** |
| right | 54231 | 95.1 | 95.3 | **95.2** |
| single | 2690 | 93.1 | 82.3 | **87.4** |

Confusion matrix:

| Reference \ predicted | 1 | 2 | 3 |
|---|---:|---:|---:|
| 1 left | **51696** | 2488 | 88 |
| 2 right | 2456 | **51700** | 75 |
| 3 single | 273 | 203 | **2214** |

Most frequent mistakes (reference → predicted): left → right (2488), right → left (2456), single → left (273), single → right (203), left → single (88), right → single (75)

### Document starts

431,203 page transitions, 1786 of them true document starts: precision 98.8, recall 99.9, F1 99.3, AP 100.0, 0.05 false splits per 1,000 non-starts. Documents recovered exactly: 1955 of 1979 (98.8%), with 1999 predicted.

| Document start | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| not a start | 429417 | 100.0 | 100.0 | **100.0** |
| start | 1786 | 98.8 | 99.9 | **99.3** |

Confusion matrix:

| Reference \ predicted | 1 | 2 |
|---|---:|---:|
| 1 not a start | **429395** | 22 |
| 2 start | 2 | **1784** |

Most frequent mistakes (reference → predicted): not a start → start (22), start → not a start (2)

Recall of the true starts by what the neighbouring documents are:

| Boundary between | True starts | Found | Recall |
|---|---:|---:|---:|
| unrelated documents | 1777 | 1775 | 99.9 |
| same volume (adjacent issues or parts) | 9 | 9 | 100.0 |

