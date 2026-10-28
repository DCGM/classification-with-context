# MetaKat evaluation-batch consistency

The check uses the R16 window-512 boundary-five model with auxiliary start/end
heads and the 0.05 fine-tuning tail. The deployed artifact and original training
checkpoint contain exactly equal model weights.

Both paths receive the same scan-batch ordering and MetaKat's detected page
numbers, filtered independently by their pagination encoders. Document boundaries
are unknown; only the beginning/end of the entire scan is supplied. True document
identities and DB pagination are not model inputs. Inference uses window 512,
overlap 64, regular tail, centered-logit merging, cubic ramp reaching full weight
at distance 48 and an ignored edge of 12 pages. These batches contain at most 399
pages, so this real-data comparison does not exercise overlapping windows; the
separate [model parity check](metakat-boundary5-inference-parity.md) covers longer
synthetic inputs.

The native path reads the original immutable FP16 training feature store. The
MetaKat path re-encodes the prepared full-resolution images and ALTO files.
Agreement between these paths is not ground-truth accuracy.

## Complete 50-batch result

All 50 local CLI batches completed successfully. Comparison covers 6,116 pages
and 6,066 eligible boundary positions (the first page of each scan has no pair
prediction).

| Output | Disagreements | Agreement |
|---|---:|---:|
| Page type | 349 / 6,116 | 94.294% |
| Side | 223 / 6,116 | 96.354% |
| Document start | 2 / 6,066 | 99.967% |

Both paths mark 29 common document starts. MetaKat marks one additional start
that the native path does not, and the native path marks one other start that
MetaKat does not. This agreement does not say whether either path agrees with
the annotated reference boundaries.

## Controlled model check

On `issues_001_nkp_b4cf0b13`, `issues_002_uzei_adfec4ad`, and
`monographs_002_mixed` (245 pages), freshly extracted MetaKat features were fed to
both independent implementations using the same FP32 feature arrays and FP16
CUDA autocast. All five heads returned exactly equal probabilities: page type,
side, pair boundary, auxiliary start, and auxiliary end. Repeating MetaKat feature
extraction also reproduced the saved CLI classifications and selected page-type
probabilities exactly on these three batches.

## Feature differences

| Batch | Pages | Mean DINO cosine | Mean mmBERT cosine |
|---|---:|---:|---:|
| issues_001_nkp_b4cf0b13 | 16 | 0.949272 | 0.994327 |
| issues_002_uzei_adfec4ad | 16 | 0.912634 | 0.988944 |
| monographs_002_mixed | 213 | 0.792183 | 0.992591 |

The statistics are exactly equal after rounding MetaKat's freshly computed
values to FP16. Pagination, presence flags and five-bit boundary inputs match
exactly on all three diagnostic batches.

The user confirmed that original thumbnails preserve aspect ratio and have a
256-pixel longest edge. This geometry is established independently of the
unrelated resizing utility. Interpolation, integer rounding, JPEG settings and
the exact decoder remain unverified. No crop or square stretch belongs in this
thumbnail stage; the separate DINO input stage resizes the thumbnail to 224x224.

The largest difference is the image input. DINO training-store metadata records
`/homes/ihradis/naki.images/public.256` as its source. The prepared batches contain
full-resolution archived images. On three pages selected for large DINO mismatch,
simulating 256-pixel long-edge thumbnails with area resizing and JPEG quality 90
increased cosine similarity from 0.2854/0.3523/0.3873 to
0.9841/0.9882/0.9482. This strongly supports a thumbnail-preprocessing explanation;
it does not establish the exact original thumbnail-generation recipe.

The text reconstruction also differs. The original dump uses
`extract_page_xml_text` in
`semAnt/librarymetadata/librarymetadata/db/parsers/parser_page_xml.py`: PAGE-XML
regions follow their reading order, with lines kept within regions. The current
MetaKat mmBERT input uses `alto_text`, which sorts all ALTO lines globally by
vertical coordinate. It can interleave columns and change which text survives
the 4,000-character limit. On the 16-page first issue batch, none of the current
ALTO prefixes equalled the original PAGE-XML prefixes; using ALTO document/block
order recovered 12/16. Re-encoding the original PAGE-XML text gave mean cosine
0.999985 (minimum 0.999939) against the stored mmBERT vectors, compared with
0.994327 for the current reconstruction. Remaining near-unit differences can
include batch/compute rounding; the exact source text and numerical conditions
were not compared byte for byte with the removed original text LMDB.

Replacing only native image features with fresh image features reduced
page-type disagreements against fresh MetaKat inference on `monographs_002_mixed`
from 68 to 4 pages. Replacing only text features left 73 disagreements. This
isolates image features as the dominant cause for that batch; it is not an
accuracy comparison or proof that one input resolution is universally better.

No production encoder settings or completed predictions were changed by this
investigation. Matching the original thumbnail preprocessing and reconstructing
text in region reading order should precede a claim of end-to-end numerical
parity with the training feature store.

## Unrelated thumbnail utility: sensitivity probe only

The checked semAnt branches contain the OCR dumper but no image resize/export
implementation. The user clarified that the following naki-tools utility is
not the exporter used for this dump. It was tested only as a preprocessing
sensitivity probe, and must not be used as historical provenance. It lives in
`/home/ikohut/Projects/naki-tools/naki_tools/data/resize_images.py` (tracked version
`f84b178` is the last commit touching that file). Its JPEG quality default is 90.
It reads with `cv2.IMREAD_UNCHANGED`, preserving grayscale, keeps aspect ratio,
uses `cv2.INTER_AREA`, rounds target dimensions, clamps them to at least one
pixel, and never enlarges smaller images. The utility's size default is 512;
256 was explicitly selected for these tests and was subsequently confirmed by
the user as the original thumbnail longest-edge size. The utility itself is
still unrelated to the original export.
Neither its JPEG quality default nor its resizing rules establish any setting
of the original image dump. The original exporter still needs to be located.

For this unrelated sensitivity probe, using max-size 256 and JPEG quality 90:

| Batch | Original full-image mean cosine | Exporter mean cosine | Exporter minimum cosine |
|---|---:|---:|---:|
| issues_001_nkp_b4cf0b13 | 0.949272 | 0.994724 | 0.986934 |
| issues_002_uzei_adfec4ad | 0.912634 | 0.993554 | 0.979059 |
| monographs_002_mixed | 0.792183 | 0.994244 | 0.948157 |

A floor-dimension variant at the same quality was also checked; it was worse on
the 213-page monograph batch (mean 0.988771). Neither variant establishes exact
thumbnail reproduction. Production preprocessing and completed batch outputs
remain unchanged. The reusable test is
`metakat_checks/probe_thumbnail_preprocessing.py`, run in MetaKat's environment
with its checkout on `PYTHONPATH`. It imports the supplied resize utility and
saves vectors and similarity statistics; temporary thumbnails are removed.

## Lower-JPEG-quality sweep

Only JPEG quality was varied; OpenCV decoding (including grayscale), aspect
preserving 256-pixel longest-edge resizing, rounded dimensions, and DINO model
preprocessing were held fixed. This remains a sensitivity probe, not provenance
for the original image exporter. All seven qualities were tested on the same
245 public pages across the three diagnostic batches.

| JPEG quality | Mean cosine to stored DINO vectors | Minimum cosine |
|---|---:|---:|
| 90 | 0.994231 | 0.948157 |
| 85 | 0.991622 | 0.941688 |
| 80 | 0.987500 | 0.918207 |
| 75 | 0.981996 | 0.880712 |
| 70 | 0.976541 | 0.874646 |
| 60 | 0.959652 | 0.811079 |
| 50 | 0.939923 | 0.719436 |

Reducing JPEG quality below 90 worsened similarity on this sample. Quality 90
is the best of these tested values, not an established historical setting.
Results: `/home/ikohut/tmp/metakat-evaluation-parity/jpeg-quality-sweep/`.
Production encoder settings and saved evaluation predictions remain unchanged.

## Direct original-encoder code comparison

The actual original functions from `Projects/MetaKat_text_embeddings` were run
on the same 16 public pages as the current MetaKat encoders, using shared loaded
models, the same runtime environment and controlled matching batch groups.

- Original DINO `load_image(path, 224)` and current `DinoEncoder.load_image`
  returned identical pixel arrays for every page.
- Original DINO `embed(..., float16, 'cls', ...)`, rounded to its FP16 storage
  format, and the current image encoder returned bitwise-identical vectors
  (maximum absolute error 0).
- The original and current `dinov2_checkpoint.py` loader files are byte-identical.
- Original mmBERT `embed_chunk` with `mean_pool` and BF16 autocast, rounded to
  FP16 storage, and the current text encoder returned bitwise-identical vectors
  (maximum absolute error 0). Both received the same 4,000-character prefixes,
  tokenization limits and eight-page batch groups.

This checks encoder computation given identical inputs, not reproduction of
historical GPU/software versions or arbitrary batch groupings. The original
image script has a default batch size of 256 versus 32 in the deployment config;
the text dump sorts much larger chunks than MetaKat's bounded groups of eight.
Such grouping changes can cause numerical rounding differences. Original text
inference consumes the already-truncated text LMDB; MetaKat applies the character
limit inside its encoder. The actual image-file and text-order differences
identified above remain upstream of this equivalent encoder computation.

Reusable test: `metakat_checks/compare_encoder_implementations.py`.
Result: `/home/ikohut/tmp/metakat-evaluation-parity/encoder-code-comparison.json`.

## Outputs

Completed native comparison:
`/home/ikohut/tmp/metakat-evaluation-parity/native-report.json` and
`disagreements.csv`, plus per-batch native probability arrays. The JSON records `state: completed`, 50 batch entries and aggregate agreement.
Diagnostic arrays, controlled model comparisons and resize/text probes are in
the same directory. Scripts are documented in
[metakat_checks/README.md](../metakat_checks/README.md#compare-deployment-with-training-inference).
