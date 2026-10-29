# MetaKat engine comparison: chapter-v1.6.0 vs chapter-v1.7.0

**chapter-v1.6.0** predicts the page type of each page alone and has no side or document-start model of its own. **chapter-v1.7.0** reads the whole batch as one scan (512-page windows) and predicts page type, side and document starts together. Both ran over the same prepared batches, with the same images and ALTO.

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
| chapter-v1.7.0 | 6033 | 89.8 | 89.9 | 54.1 |

chapter-v1.7.0 fixes 962 pages and breaks 226 (paired McNemar p = 2e-108).
On the 203 curated pages alone: chapter-v1.6.0 accuracy 76.8, chapter-v1.7.0 92.1 (fixed 35, broken 4).

#### Per page type

Precision (P), recall (R) and F1 per engine; *fixed*/*broken* count the pages of that reference class that chapter-v1.7.0 gets right or wrong where chapter-v1.6.0 did not or did.

| Page type | Support | chapter-v1.6.0 P | R | F1 | chapter-v1.7.0 P | R | F1 | Fixed | Broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advertisement | 106 | 19.9 | 70.8 | 31.1 | 75.0 | 53.8 | 62.6 | 17 | 35 |
| appendix | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| backCover | 77 | 37.4 | 71.4 | 49.1 | 68.6 | 62.3 | 65.3 | 8 | 15 |
| backEndPaper | 35 | 25.0 | 28.6 | 26.7 | 35.7 | 28.6 | 31.7 | 8 | 8 |
| backEndSheet | 73 | 16.9 | 39.7 | 23.7 | 70.8 | 69.9 | 70.3 | 33 | 11 |
| blank | 390 | 74.9 | 54.4 | 63.0 | 77.8 | 74.6 | 76.2 | 136 | 57 |
| calibrationTable | 55 | 100.0 | 90.9 | 95.2 | 42.4 | 96.4 | 58.9 | 4 | 1 |
| cover | 22 | 100.0 | 90.9 | 95.2 | 0.0 | 0.0 | 0.0 | 0 | 20 |
| dedication | 5 | 100.0 | 40.0 | 57.1 | 40.0 | 80.0 | 53.3 | 3 | 1 |
| edge | 60 | 100.0 | 21.7 | 35.6 | 79.6 | 71.7 | 75.4 | 33 | 3 |
| flyleaf | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| frontCover | 76 | 47.0 | 81.6 | 59.6 | 78.9 | 78.9 | 78.9 | 7 | 9 |
| frontEndPaper | 35 | 34.8 | 22.9 | 27.6 | 53.3 | 22.9 | 32.0 | 7 | 7 |
| frontEndSheet | 73 | 31.1 | 19.2 | 23.7 | 81.7 | 67.1 | 73.7 | 39 | 4 |
| frontispiece | 4 | 40.0 | 50.0 | 44.4 | 7.0 | 75.0 | 12.8 | 1 | 0 |
| illustration | 172 | 85.5 | 79.1 | 82.2 | 93.5 | 83.1 | 88.0 | 17 | 10 |
| imprimatur | 1 | 0.0 | 0.0 | 0.0 | 7.7 | 100.0 | 14.3 | 1 | 0 |
| index | 45 | 35.0 | 62.2 | 44.8 | 56.7 | 75.6 | 64.8 | 12 | 6 |
| jacket | 2 | 33.3 | 100.0 | 50.0 | 40.0 | 100.0 | 57.1 | 0 | 0 |
| normalPage | 3890 | 97.7 | 81.4 | 88.8 | 96.8 | 96.1 | 96.4 | 581 | 12 |
| preface | 8 | 37.5 | 75.0 | 50.0 | 0.0 | 0.0 | 0.0 | 0 | 6 |
| sheetMusic | 479 | 96.8 | 100.0 | 98.4 | 92.1 | 100.0 | 95.9 | 0 | 0 |
| spine | 23 | 100.0 | 4.3 | 8.3 | 91.7 | 47.8 | 62.9 | 11 | 1 |
| table | 70 | 28.2 | 70.0 | 40.2 | 78.6 | 78.6 | 78.6 | 10 | 4 |
| tableOfContents | 42 | 45.5 | 47.6 | 46.5 | 78.8 | 61.9 | 69.3 | 10 | 4 |
| titlePage | 283 | 79.7 | 84.5 | 82.0 | 90.0 | 88.7 | 89.3 | 24 | 12 |

#### Most frequent page-type errors (reference → predicted)

- **chapter-v1.6.0:** normalPage → advertisement (293), normalPage → table (124), blank → backEndSheet (83), normalPage → index (48), normalPage → titlePage (47), normalPage → listOfIllustrations (45), normalPage → flyleaf (44), blank → flyleaf (31), titlePage → frontCover (22), backEndSheet → blank (20)
- **chapter-v1.7.0:** advertisement → normalPage (37), blank → calibrationTable (28), illustration → normalPage (25), normalPage → titlePage (23), normalPage → blank (22), blank → normalPage (20), normalPage → index (19), backEndPaper → blank (18), normalPage → advertisement (18), blank → sheetMusic (18)

### Side, all pages together (108 pages with a DB placement)

chapter-v1.6.0 has no side prediction. Precision is computed over the pages that have a reference placement only.

| Side | Support | chapter-v1.7.0 P | R | F1 |
|---|---:|---:|---:|---:|
| left | 50 | 78.0 | 78.0 | 78.0 |
| right | 50 | 77.6 | 76.0 | 76.8 |
| singlePage | 8 | 66.7 | 75.0 | 70.6 |

Overall side accuracy of chapter-v1.7.0: 76.9. Confusion (reference → predicted): left → left: 39, left → right: 10, left → singlePage: 1, right → left: 10, right → right: 38, right → singlePage: 2, singlePage → left: 1, singlePage → right: 1, singlePage → singlePage: 6

### Document starts, by kind of batch

The long `boundaries/*` batches hold many short documents next to each other: `issues` (periodical issues of one volume), `books` (unrelated short books) and `mixed`. The others are the regular groups: `issues` and `multipart` batches of two or three adjacent documents, `monographs` of unrelated standalone books, `single` of one document (no true starts to find, so every predicted start there is a false alarm).

| Batch kind | Batches | Pages scored | True starts | chapter-v1.6.0: found | predicted | P | R | F1 | chapter-v1.7.0: found | predicted | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 59 | 9110 | 245 | 131 | 267 | 49.1 | 53.5 | 51.2 | 222 | 233 | 95.3 | 90.6 | 92.9 |
| boundaries/books | 3 | 1194 | 48 | 31 | 65 | 47.7 | 64.6 | 54.9 | 39 | 44 | 88.6 | 81.2 | 84.8 |
| boundaries/issues | 3 | 699 | 71 | 49 | 52 | 94.2 | 69.0 | 79.7 | 71 | 71 | 100.0 | 100.0 | 100.0 |
| boundaries/mixed | 3 | 1151 | 90 | 33 | 49 | 67.3 | 36.7 | 47.5 | 86 | 87 | 98.9 | 95.6 | 97.2 |
| issues | 10 | 340 | 10 | 8 | 8 | 100.0 | 80.0 | 88.9 | 10 | 10 | 100.0 | 100.0 | 100.0 |
| monographs | 10 | 2875 | 16 | 7 | 43 | 16.3 | 43.8 | 23.7 | 10 | 11 | 90.9 | 62.5 | 74.1 |
| multipart | 10 | 1809 | 10 | 3 | 25 | 12.0 | 30.0 | 17.1 | 6 | 10 | 60.0 | 60.0 | 60.0 |
| single | 20 | 1042 | 0 | 0 | 25 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0.0 | 0.0 |

## Consistency set (fit documents: not held-out)

5341 pages in 50 batches; 278 pages have no reference page type and are not scored for it.

### Page type, all pages together

| | Pages | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---:|
| chapter-v1.6.0 | 5063 | 89.2 | 92.4 | 35.7 |
| chapter-v1.7.0 | 5063 | 94.1 | 95.6 | 52.2 |

chapter-v1.7.0 fixes 336 pages and breaks 89 (paired McNemar p = 7.8e-35).
On the 6 curated pages alone: chapter-v1.6.0 accuracy 66.7, chapter-v1.7.0 100.0 (fixed 2, broken 0).

#### Per page type

Precision (P), recall (R) and F1 per engine; *fixed*/*broken* count the pages of that reference class that chapter-v1.7.0 gets right or wrong where chapter-v1.6.0 did not or did.

| Page type | Support | chapter-v1.6.0 P | R | F1 | chapter-v1.7.0 P | R | F1 | Fixed | Broken |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| advertisement | 38 | 20.4 | 89.5 | 33.2 | 77.8 | 18.4 | 29.8 | 0 | 27 |
| backCover | 14 | 31.0 | 64.3 | 41.9 | 68.4 | 92.9 | 78.8 | 4 | 0 |
| backEndSheet | 14 | 27.3 | 42.9 | 33.3 | 76.5 | 92.9 | 83.9 | 7 | 0 |
| blank | 60 | 50.0 | 56.7 | 53.1 | 61.6 | 88.3 | 72.6 | 21 | 2 |
| frontCover | 15 | 44.4 | 80.0 | 57.1 | 47.8 | 73.3 | 57.9 | 2 | 3 |
| frontEndSheet | 16 | 0.0 | 0.0 | 0.0 | 63.2 | 75.0 | 68.6 | 12 | 0 |
| frontJacket | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| index | 3 | 4.8 | 100.0 | 9.2 | 2.7 | 100.0 | 5.3 | 0 | 0 |
| jacket | 1 | 0.0 | 0.0 | 0.0 | 16.7 | 100.0 | 28.6 | 1 | 0 |
| map | 1 | 14.3 | 100.0 | 25.0 | 0.0 | 0.0 | 0.0 | 0 | 1 |
| normalPage | 4793 | 99.7 | 90.4 | 94.8 | 99.4 | 95.0 | 97.2 | 277 | 55 |
| table | 9 | 17.8 | 88.9 | 29.6 | 30.8 | 88.9 | 45.7 | 0 | 0 |
| tableOfContents | 16 | 42.9 | 56.2 | 48.6 | 73.7 | 87.5 | 80.0 | 5 | 0 |
| titlePage | 82 | 65.1 | 84.1 | 73.4 | 74.3 | 91.5 | 82.0 | 7 | 1 |

#### Most frequent page-type errors (reference → predicted)

- **chapter-v1.6.0:** normalPage → advertisement (131), normalPage → flyleaf (68), normalPage → index (58), normalPage → table (37), normalPage → titlePage (30), normalPage → listOfIllustrations (30), normalPage → blank (29), normalPage → backCover (12), normalPage → illustration (11), normalPage → tableOfContents (10)
- **chapter-v1.7.0:** normalPage → index (107), normalPage → blank (29), advertisement → normalPage (23), normalPage → titlePage (23), normalPage → table (18), normalPage → errata (9), normalPage → illustration (6), normalPage → frontispiece (6), normalPage → impressum (6), normalPage → frontCover (4)

### Side, all pages together (246 pages with a DB placement)

chapter-v1.6.0 has no side prediction. Precision is computed over the pages that have a reference placement only.

| Side | Support | chapter-v1.7.0 P | R | F1 |
|---|---:|---:|---:|---:|
| left | 121 | 100.0 | 98.3 | 99.2 |
| right | 121 | 99.2 | 99.2 | 99.2 |
| singlePage | 4 | 50.0 | 75.0 | 60.0 |

Overall side accuracy of chapter-v1.7.0: 98.4. Confusion (reference → predicted): left → left: 119, left → singlePage: 2, right → right: 120, right → singlePage: 1, singlePage → right: 1, singlePage → singlePage: 3

### Document starts, by kind of batch

The long `boundaries/*` batches hold many short documents next to each other: `issues` (periodical issues of one volume), `books` (unrelated short books) and `mixed`. The others are the regular groups: `issues` and `multipart` batches of two or three adjacent documents, `monographs` of unrelated standalone books, `single` of one document (no true starts to find, so every predicted start there is a false alarm).

| Batch kind | Batches | Pages scored | True starts | chapter-v1.6.0: found | predicted | P | R | F1 | chapter-v1.7.0: found | predicted | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 50 | 5291 | 40 | 21 | 62 | 33.9 | 52.5 | 41.2 | 28 | 31 | 90.3 | 70.0 | 78.9 |
| issues | 10 | 296 | 15 | 10 | 14 | 71.4 | 66.7 | 69.0 | 13 | 13 | 100.0 | 86.7 | 92.9 |
| monographs | 10 | 2646 | 15 | 9 | 30 | 30.0 | 60.0 | 40.0 | 10 | 11 | 90.9 | 66.7 | 76.9 |
| multipart | 10 | 2082 | 10 | 2 | 18 | 11.1 | 20.0 | 14.3 | 5 | 6 | 83.3 | 50.0 | 62.5 |
| single | 20 | 267 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 0.0 | 0.0 | 0.0 |


Generated by `python -m metakat_checks.compare_engines --output <dir>` from the batch manifests and the `<batch>.<engine>.results` directories.
