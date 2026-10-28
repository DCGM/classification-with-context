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

## Image mismatch by reference page type

The cached image vectors were grouped by human annotation where available,
otherwise DB page type. Unlabelled pages remain separate; contextual predictions
are not treated as reference labels. This is the same 245-page diagnostic sample,
not all 6,116 evaluation pages. It includes no reference-labelled TOC pages.
The thumbnail column uses aspect-preserving max-size 256 and JPEG quality 90
with rounded dimensions and the controlled unrelated utility described above.

| Reference type | Pages | Full-resolution mean cosine | Thumbnail mean cosine | Thumbnail minimum |
|---|---:|---:|---:|---:|
| Edge | 9 | 0.545867 | 0.980638 | 0.948157 |
| Spine | 3 | 0.548331 | 0.982983 | 0.964659 |
| FrontEndPaper | 4 | 0.970310 | 0.989770 | 0.969466 |
| Preface | 2 | 0.617033 | 0.990982 | 0.983096 |
| TitlePage | 7 | 0.928667 | 0.993700 | 0.984370 |
| FrontCover | 3 | 0.968781 | 0.993885 | 0.989786 |
| Dedication | 2 | 0.848510 | 0.994080 | 0.993989 |
| NormalPage | 26 | 0.931437 | 0.994530 | 0.979059 |
| Unlabelled | 153 | 0.761585 | 0.994937 | 0.980324 |
| Cover | 1 | 0.983285 | 0.995040 | 0.995040 |
| Frontispiece | 1 | 0.976505 | 0.995132 | 0.995132 |
| Blank | 15 | 0.973102 | 0.995287 | 0.982873 |
| BackEndSheet | 3 | 0.966997 | 0.995312 | 0.993865 |
| CalibrationTable | 4 | 0.970729 | 0.995371 | 0.986896 |
| Advertisement | 2 | 0.955040 | 0.996840 | 0.994874 |
| BackCover | 3 | 0.977141 | 0.997497 | 0.996267 |
| BackEndPaper | 4 | 0.983481 | 0.998276 | 0.997540 |
| FrontEndSheet | 3 | 0.964248 | 0.998591 | 0.997998 |

The strongest remaining mismatch is on Edge and Spine pages. The small per-type
sample counts prevent a population-level conclusion. The per-page CSV and the
20 largest residuals are saved under
`/home/ikohut/tmp/metakat-evaluation-parity/image-types/`.

## Crop-history audit

The user recalled reproducing the DINO training loader's center-crop behavior
for thin spines. The earlier encoder-equality check did not test that training
loader; it only compared the current dump script with the current MetaKat port.

The tracked dump inference script was introduced in `b82c008` on 2026-09-29
with whole-image bilinear resizing and documentation explicitly saying
"without its random crops". The next change, `ef0e763` the same day, replaced
bilinear interpolation with `cv2.INTER_AREA`; it did not remove a center crop.
The MetaKat context port introduced in `3b47119` on 2026-10-26 copied this
whole-image path. Subsequent relocation changed imports, not image geometry.
This history shows an explicit choice to omit crops when the tracked dump
script was introduced, but establishes no later removal of the user's verified
center-crop implementation and does not identify that implementation.

The original DINO run's `train.log` identifies LightlyTrain 0.17.0 and records
random-resize global views with scale range 0.32..1.0 and local views with
scale range 0.05..0.32. That differs from deterministic whole-image extraction;
this log alone does not reproduce the precise crop implementation or fallback.
The cached public image-vector metadata itself declares whole-image 224x224
`INTER_AREA` extraction, not center cropping.

As an additional diagnostic, three plausible central crops were applied to the
same simulated 256/Q90 thumbnails on the reference-labelled spines and edges:

| Input variant | Spine mean cosine (3 pages) | Edge mean cosine (9 pages) |
|---|---:|---:|
| Whole thumbnail to 224x224 | 0.982983 | 0.980639 |
| Central square to 224x224 | 0.700661 | 0.600998 |
| Central aspect-ratio-constrained crop | 0.734112 | 0.635836 |
| Short-edge-256 / center-224 equivalent extent | 0.664797 | 0.586800 |

These are crop sensitivity probes, not an exact reproduction of the original
training transform. They support whole-image extraction as the source of these
stored contextual-training vectors. A DINO-pretraining-versus-extraction mismatch
is distinct from MetaKat-versus-context-training-store parity. Production image
inference was not changed during this audit.

Audit data: `/home/ikohut/tmp/metakat-evaluation-parity/center-crop-probe.json`.

## Full dump-script history and historical-loader replay

Fetched the shared MetaKat repository and checked all branches, reflogs and
107 unreferenced recoverable commits. None of the unreferenced commits contained
another snapshot of the DINO dump script at its historical path. The separate
`65d65da` thin-image fix concerns Gaussian-blur padding in an older classifier
dataset; it does not change DINO embedding inference.

| Commit | Date | Change | Changes pixel preparation? |
|---|---|---|---|
| `b82c008` | 2026-09-29 | Introduce DINO export loader, full-page PIL bilinear resize, JPEG draft decoding | Initial behavior |
| `ef0e763` | 2026-09-29 | PIL bilinear -> OpenCV INTER_AREA | Yes |
| `4b87327` | 2026-10-02 | Find images inside per-document directories | No, changes file discovery |
| `8a2d3e2` | 2026-10-02 | Key vectors by image-relative path instead of library/page UUID | No, changes identity/keying |
| `13546fa` | 2026-10-06 | LMDB map-size default | No |
| `3b47119` | 2026-10-26 | Introduce context-engine DINO port using whole-page INTER_AREA | Same image computation |
| `8cf98b7` | 2026-10-27 | Remove old training CLI and relocate checkpoint loader | No |

The first dump-script version already uses FP16 CUDA compute by default, CLS
pooling and SDPA. No tracked change replaces center cropping with whole-page
resizing in this script. A different verified loader or an uncommitted original
server edit is still possible; neither is established by this Git history.

The original public dump's `log.txt` starts at 2026-10-02 14:24:39 UTC and records
`/homes/ihradis/metakat/MetaKat/.../infer_dinov2_vitb14.py`, source
`/homes/ihradis/naki.images/public.256/`, Torch 2.8.0+cu128 and Transformers 4.57.6.
This is after both October 2 file/keying commits, accounting for their +0200 Git
timestamps. The log does not record a source-file hash or prove that the remote
working tree was clean.

The actual `load_image` function from Git revisions `b82c008` and `ef0e763` was
replayed on identical simulated 256/Q90 thumbnails. The current DINO model and
pooling were held fixed:

| Historical loader | Mean cosine, 213-page batch | Spine mean (3 pages) | Edge mean (9 pages) |
|---|---:|---:|---:|
| PIL bilinear (`b82c008`) | 0.987983 | 0.875050 | 0.893088 |
| INTER_AREA (`ef0e763`) | 0.994244 | 0.982983 | 0.980638 |

The old interpolation explains substantial sensitivity on thin pages but does
not improve reproduction of the cached vectors in this probe. Central cropping
also worsens it, as shown above. These tests use reconstructed thumbnails, so
none settles the historical file-level preprocessing exactly.

To exclude the shared-store migration and key mapping as causes, all 245 native
raw image vectors were compared against the original public image LMDB using
each manifest's exact relative-path key. All exist and are bitwise identical
(maximum absolute difference 0).

The most useful next comparison is an actual 256-pixel spine/edge JPEG used by
the original dump. Using the original file eliminates thumbnail reconstruction
uncertainty. If the original file reproduces the cached vector, investigate
thumbnail generation/source-image differences; if it does not, investigate the
original server's code snapshot and runtime (including Torch 2.8 versus current
Torch 2.14) before changing production preprocessing.

Artifacts: `historical-loader-probe.json` and
`native-vs-original-image-lmdb.json` under the parity work directory.

## Checkpoint and source revision confirmed by original dump machine

The user supplied the original machine's checkpoint MD5:
`0eb8baa6b8de4c4ab94c211aaf632c83`, and Git revision
`8a2d3e2d334132ff0b85e9495be574d782687037`.

Both the SSD source checkpoint and the DINO checkpoint in the MetaKat v1.7.0
bundle have exactly that MD5 and size 346,396,515 bytes. Their SHA256 is
`d08f6d05da58efff5748274f487c2583d94cad5ee9e0490a57e40c3a3832f80e`,
matching the training-vector metadata and bundle configuration. This excludes
selection of a different exported checkpoint as the cause.

Comparing the local `MetaKat_text_embeddings` image-inference script against the
supplied commit shows only the LMDB map-size default changing from 100 to 300 GB;
its image preparation and embedding computation do not differ. The checkpoint
loader file is unchanged. The user-supplied Git revision identifies the original
checkout HEAD, without independently establishing absence of uncommitted changes.

## 224-pixel thumbnail probe

Tested 224 instead of 256 as the aspect-preserving thumbnail longest edge on
the same 245 pages, keeping JPEG quality 90, decoding, interpolation, dimension
rounding and DINO's final 224x224 preparation unchanged.

| Thumbnail longest edge | Pages | Mean cosine | Minimum cosine |
|---|---:|---:|---:|
| 256 | 245 | 0.994231 | 0.948157 |
| 224 | 245 | 0.851737 | 0.580920 |

224-pixel thumbnails are a worse match to stored DINO vectors on this sample.
Production settings and completed predictions remain unchanged. Results are in
`/home/ikohut/tmp/metakat-evaluation-parity/thumbnail-224-q90/`.

## Actual original thumbnails from Sophie2: encoder parity established

The user copied the 48 selected original JPEGs from
`sophie2:/homes/ikohut/naki.images/public.256/`, preserving exact LMDB-relative
paths. All local SHA256 checksums passed. Feeding these original files directly
to the current MetaKat DINO encoder reproduced the stored training vectors:

| Measure | Result |
|---|---:|
| Pages | 48 |
| Mean cosine | 0.999999893 |
| Minimum cosine | 0.999999593 |
| Raw FP16 vectors bitwise identical | 32 / 48 |
| Spine raw FP16 vectors bitwise identical | 3 / 3 |
| Edge raw FP16 vectors bitwise identical | 5 / 9 |
| Mean spine cosine | 0.999999979 |
| Mean edge cosine | 0.999999889 |

The residual on the remaining vectors is at small numerical-rounding scale.
This establishes inference parity on actual original inputs and removes the
need to change crop policy or the model to explain the earlier mismatches.
Those mismatches arise when reconstructing thumbnails from the archived
full-resolution images, not when embedding the originals.

The original files also establish the following JPEG properties:

- All 48 have a 256-pixel longest edge and RGB mode.
- All 48 have quantization tables exactly matching standard JPEG quality 90.
- All 48 use 4:2:0 chroma subsampling, matching the Q90 reconstructions.
- Only 26/48 reconstructed thumbnails have the same dimensions as the originals.
  Original dimensions agree with a direct floor rule on 23/48 pages, round on
  26/48, and ceil on 25/48; none of these simple direct rules explains every page.
- Every dimension mismatch is exactly one pixel on the shorter edge: 11
  originals are one pixel wider, nine are one pixel narrower, one is one pixel
  taller, and one is one pixel shorter. The 256-pixel longest edge matches in
  all 48 cases. Examples: original spine 32x256 versus reconstruction 31x256;
  another spine 36x256 versus 35x256; edge 42x256 versus 43x256; title page
  186x256 versus 185x256. Thus these are small integer-geometry differences,
  not gross scaling failures. The exact originating rounding/intermediate
  resize rule is still unknown.
- Even dimension-matched examples have differing decoded pixels. Therefore
  thumbnail generation/source-image/decoder details remain to be identified;
  JPEG quality alone and a single direct rounding-rule change are insufficient.

Reusable check: `metakat_checks/compare_original_thumbnails.py`.
Report and vectors:
`/home/ikohut/tmp/metakat-evaluation-parity/original-thumbnail-check/`.
Originals: `/home/ikohut/tmp/metakat-evaluation-parity/original-thumbnails/`.
Production encoder settings and completed evaluation predictions remain unchanged.

## Outputs

Completed native comparison:
`/home/ikohut/tmp/metakat-evaluation-parity/native-report.json` and
`disagreements.csv`, plus per-batch native probability arrays. The JSON records `state: completed`, 50 batch entries and aggregate agreement.
Diagnostic arrays, controlled model comparisons and resize/text probes are in
the same directory. Scripts are documented in
[metakat_checks/README.md](../metakat_checks/README.md#compare-deployment-with-training-inference).
