# Experiments: tagging streams of scanned pages

Status: experiment plan with the first modular implementation in
[page_sequence](../page_sequence/README.md). Numerical settings below are starting points to
validate, not measured optima. Synthetic training and pretrained BERT-Mini forward/backward
checks are available; data-order and loader audits have been performed; corpus quality training has not started.

## Task and assumptions

Input is an ordered stream of page vectors, potentially spanning several documents. Output,
aligned one-to-one with the input pages:

- page type: the existing 39-class label space;
- placement: left / right / single;
- probability that a new document starts before this page.

Model outputs are page-aligned: type_logits (B,L,39), placement_logits (B,L,3), and
starts_document_logits (B,L). starts_document[i] belongs to page i; its head may use both
h[i-1] and h[i]. Keep a validity mask per task, independent of padding. The first window position's
start target is masked unless a preceding halo supplies its predecessor; the corresponding
numerical logit is a placeholder, not an inferred negative. Missing type/placement/start labels
mask only their direct loss. Such pages still provide context and receive gradients through
other valid predictions. Missing placement is not single. Return labels/probabilities and segment
IDs in input-page order after full-stream merging, retaining unknown first-page start status.

Initial assumption: documents appear consecutively and each image represents one page. Page
reordering, splitting double-page images and merging fragments are separate tasks. The document
unit is the lowest document level in the DB, represented by (library, doc) in the CSV. For the
periodical -> volume -> issue hierarchy, every issue is a distinct document, including consecutive
issues in the same volume. This target definition is settled; root/volume boundaries do not
replace issue boundaries.

One page becomes one sequence token after feature fusion. Batch length is measured in pages,
not OCR tokens. Freeze the existing text and image embedding extractors in these experiments.

Use the boundary between pages i-1 and i as the primary binary target. An internal chunk's first
page has no visible predecessor: mask this boundary loss unless a preceding halo provides it.
For a complete scan batch, open the first output segment by convention. For visible predecessor
pairs the target is simply starts_document[i] = int(doc_key[i] != doc_key[i-1]). Document IDs are
used by the sampler to construct these targets, never supplied as input features.

## Available supervision and its limits

The README describes 45,056,755 training-context rows, 24,442 annotated training pages,
493,714 test-context rows, and 2,536 annotated test pages. Existing probes use known-document
windows; these are useful references but give information unavailable in the new deployment.

The annotation column is curated supervision for page type. DB page types are weak labels:
useful at scale, with class-dependent errors and very heavy NormalPage dominance. Empty and
unsupported DB labels have no type loss; they still contribute to other applicable objectives.
Where an annotation exists, use it instead of a conflicting DB type. Do not silently map extra
DB classes; keep the mapping explicit and versioned.

Placement is DB metadata, including in the annotated CSVs. Its quality and coverage require an
audit; a page-type annotation does not imply a manually checked placement label. Empty placement
is an unknown target to mask, not the class single. Synthetic document-start labels are obtained
directly from leaf-document identity transitions and are exact for the constructed stream.

The pruned dump excludes pages without the required embeddings/text. A retained document may
therefore be incomplete; record this as a data condition. A join from one retained document's
end to another's beginning is still a positive boundary under the chosen task. An order gap
inside the same document is not a boundary, and an arbitrary window crop is not a new document.

## Split before sampling

1. Preserve the existing test documents as the final test partition. Exclude their context from
   all training, including weak-label training, self-supervision and normalization fitting.
2. Hold out approximately 15% of training document groups for development. Exclude their whole
   context from training, not just their annotated rows. Keep tiny classes represented in fit
   where possible, and record classes for which reliable development estimates are impossible.
3. Group documents sharing page UUIDs/known copies before splitting. Check overlap with the
   existing test partition and quarantine conflicting training groups; do not move test pages
   into training. Root/parent groups can also support a stricter generalization evaluation.
4. Construct synthetic streams independently within each split. Freeze development/test stream
   manifests and seeds. Never choose context length, thresholds or checkpoints on test results.

The available text/image extractors were computed previously. Record their training provenance;
document separation here cannot retrospectively guarantee they never saw test images.

## Feature normalization and fusion

Initial inputs: DINOv2 (768), mmBERT (768), and a grouped page-statistics view. Compare Qwen3
(1024) as a replacement for mmBERT before paying for both text models in every experiment.

For each dense embedding modality independently, decode to float32 and compute an epsilon-safe
L2 normalization. Zero missing vectors and supply explicit modality-presence flags. A real blank
page and a missing feature vector are different cases. Use one learned projection of the
concatenated inputs initially, without separate learned modality branches.

For page statistics, begin with all grouped features, including header/footer keywords, patterns,
ALTO geometry, word confidence, word boxes and availability flags. Do not set include or exclude
for the starting model. Use PageStats.view(full, grouped=True, normalize=True) for fixed bounded
normalization, or pass the training-fitted normalizer to normalize. Learned fusion can suppress
unhelpful features; validate this rather than assuming nuisance features cannot hurt. Feature
removal, including the word-confidence/word-box blocks, is a later ablation. Respect view()'s
existing log transforms; do not log twice. The initial model uses the implemented bounded [0,1]
normalizer described below. If fitting scales, use positive values from a reproducible training-only
sample and freeze the result for inference. Source-missing zeros remain zero, and availability
flags remain present. Invalid/nonfinite inputs are audited and repaired or explicitly quarantined
before normalization; the implementation raises rather than silently replacing them. Compare
featurewise clipped standardization and the full per-stem view only as later ablations.

Concatenate the preprocessed modalities and presence flags; project the result to d_model=256.
Add sequence-position information once per page. No cross-page concatenation at this stage.
Modality identity is implicit in the fixed coordinate blocks of the concatenation.

### Concrete first configuration for BERT-Mini

    image[768] -> safe L2 -> zero if missing
    text[768]  -> safe L2 -> zero if missing
    stats[S]   -> fixed feature transforms -> divide by sqrt(S) -> zero missing blocks

    concat(image768, text768, statsS, availability flags)
        -> Linear(1536 + S + n_flags,256) -> LayerNorm -> learned scalar scale
        -> BERT-Mini inputs_embeds

S and the feature order come from a versioned grouped PageStats view; availability flags include
missing modalities and relevant unavailable source blocks. No additional z-scoring or [-5,5]
clipping is applied on top of bounded normalization in the initial model.
The fixed 1/sqrt(S) factor stops a wide bounded statistics block from
having a much larger initial norm than the unit-norm dense modalities. It retains variation in
statistics magnitude between pages, unlike per-page L2 normalization of that block. It is an
initial scale convention, not a guarantee of equal modality importance; compare a learned scalar
gain per block if needed. Missing inputs become zero after fixed preprocessing, before fusion.

The fusion scale is initialized from a typical centered per-coordinate RMS of the checkpoint's
word embedding rows, measured once when loading, and then learned. This is a proposed scale
calibration, not a guarantee of distribution matching. BERT adds its pretrained position and
constant segment-0 embeddings, then applies its existing embedding LayerNorm/dropout. That last
LayerNorm does not recover a useful content/position ratio if page inputs initially dwarf the
position embeddings. Our fusion LayerNorm stabilizes content scale before that addition. Do not
add a second external position vector or supply true document IDs through token_type_ids.
The original implementation establishes this order:
[BertEmbeddings](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/bert/modeling_bert.py#L120-L175).

L2 normalization operates over each page's dense vector; statistics normalization uses fixed
defaults or saved training-fitted scales; LayerNorm operates over the channels of one page representation. None
uses the current stream's mean/std or normalizes across page positions. Avoid BatchNorm initially:
mixed annotation/dump batches and different inference chunk compositions would complicate it.

One linear projection already has independent weight blocks for each modality:

    W * concat(image, text, stats) = W_image * image + W_text * text + W_stats * stats

Separate linear branches followed by linear fusion collapse to this form, potentially with rank
constraints from their intermediate bottlenecks. They are not inherently more expressive. Per-
branch LayerNorm or nonlinearities do change the function, and branch bottlenecks change capacity
and regularization; these require experimental justification. The padding attention mask remains
separate from modality-presence flags. Real blank-page features remain valid inputs.

Keep fusion ablations small:

- direct concatenation and one projection (initial recommendation);
- one joint MLP instead of the single linear projection, to test nonlinear fusion;
- separate nonlinear/normalized branches, only as a later capacity-controlled comparison; mask
  missing branch outputs after their biased operations if using this variant;
- training-set featurewise standardization of dense embeddings instead of L2 normalization.

Potential norm information can be tested as a separate log-norm scalar rather than entangled
with feature scale. Fit every transform on training only and save it with the checkpoint.
Use modest modality dropout, initially 0.1, only after the no-dropout baseline; monitor missing-
modality slices explicitly. Keep page positions even when a feature modality is missing.

### Bounded statistics normalization

Start with the implemented feature-aware [0,1] representation before the statistics MLP or joint
projection; clipped standardization is an alternative to test later. The implementation mixes already logged values,
flags, proportions, and unbounded measurements; a blanket log or min/max transform is unsuitable.

- Keep binary flags and valid proportions/shares in [0,1]. Audit nominal geometry bounds rather
  than treating malformed coordinates as reliable proportions.
- For nonnegative counts and long-tailed measurements, first define g_j(x): log1p(x) for raw
  counts/lengths/gaps, or identity for features already logged by the extractor/view. Then use
  u_j = g_j(x)/(g_j(x) + s_j). Fit s_j to the median positive transformed value over valid training
  rows, with an explicit semantic floor/fallback when positives are absent or the scale is tiny.
  This maps zero to zero, the fitted positive scale to 0.5, and large values smoothly toward one.
- Particularly small geometry measurements, such as relative line height, may use this same
  positive-scale transform with identity g instead of retaining tiny input amplitudes. Distinguish
  these from shares where the original [0,1] units are already useful.
- Fit scales on positive values for zero-heavy features: a whole-column median/IQR/quantile may
  be zero and is not a usable divisor. Record positive support and use predefined fallbacks for
  rare or unseen features. Do not standardize or rescale binary flags by rarity.
- Audit nonfinite or semantically invalid values before transformation, retaining availability
  flags. The normalizer rejects negative/nonfinite inputs and clips upper overshoots of proportion
  features. Establish a data-repair/quarantine policy from audit counts rather than silently
  skipping problematic pages at training time. These transforms do not fix upstream division by zero.

All positive-scale choices are proposed settings, not measured distributions. Save feature names,
transform types and fitted scales. No per-page L2 normalization of the whole statistics vector:
it would make one keyword's value depend on unrelated features on that page. Compare the bounded
arm on the same streams and model budget; retain the alternative clip(z,-5,5)/5 when testing a
strictly bounded [-1,1] standardized representation. Neither uses scan/window-dependent statistics.

Forbidden model inputs in the initial deployment experiment: document UUID, root/parent UUID,
library ID, file path, DB page type, placement target, document-relative order/rank, document
length, boundary masks and positional resets at known document boundaries. IDs/order may build
training sequences but are not features. Printed page numbers are a later option only if derived
the same way from OCR at inference, rather than obtained from document metadata.

## Models, from small to contextual

| Model | Initial configuration | Purpose |
|---|---|---|
| Page MLP + pair boundary MLP | Same fusion; boundary on adjacent fused page pairs | Separates page evidence from context benefit |
| Temporal CNN | Four residual width-3 layers, dilations 1/2/4/8, width 256 | Parallel local-context baseline; receptive field 31 pages |
| Bidirectional Transformer | Four pre-norm layers, width 256, 4 heads, FFN 1024, dropout 0.1 | Initial model for longer context |

Give every model a page-type softmax and a placement softmax. Predict boundaries with a small
MLP on [h(i-1), h(i), abs(h(i)-h(i-1)), h(i-1)*h(i)], yielding one logit per adjacent pair.
Use boundary probabilities to create segment IDs after tagging. Full bidirectional attention
may cross unknown document boundaries: that is intentional. Only mask padding and unrelated
training examples, never provide true-document attention masks.

Start the Transformer with sinusoidal positions and train/evaluate at supported lengths; do not
assume good extrapolation to much longer streams. Relative-position alternatives can follow.

Keep independent heads first. If errors show implausible output transitions, try learned CRF
decoding, starting with placement. Do not hard-code left/right alternation: blanks, inserts and
missing pages can break it. Type transitions should be boundary-aware and learned, without
enforcing a universal book template. Sparse type supervision would require marginalizing unknown
labels in a CRF; do not invent labels to make its training convenient.

An encoder-decoder is a later controlled experiment if output dependence helps. Constrain it to
one aligned output tuple per page and compare against CRF decoding. Autoregressive decoding adds
latency and exposure to previous prediction errors; it is not required to tag many pages at once.

## Pretrained sequence encoders

There are two distinct sources of pretraining: the frozen page feature extractors already used
by every experiment, and a possible pretrained encoder above those page features. Starting the
second encoder randomly still benefits from the first. A text encoder's word-token training is
not page-stream training; transfer to sequences of multimodal page vectors is a hypothesis.

Shortlist, based on the publishers' model cards/configs:

| Checkpoint | Published architecture | Role in these experiments |
|---|---|---|
| [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) | BERT-Mini: 4 layers, width 256, 4 heads | First small pretrained arm |
| [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 22 layers, width 384, about 42M non-embedding parameters; 8192-token context | First modern, multilingual pretrained arm if pilot resources allow |
| [microsoft/deberta-v3-xsmall](https://huggingface.co/microsoft/deberta-v3-xsmall) | 12 layers, width 384, about 22M backbone parameters | Alternative compact encoder with disentangled attention |
| [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 22 layers, width 768; 8192-token context and alternating local/global attention | Larger modern reference, after the small arms |

These context limits refer to their original text-token setting. One projected page occupies one
input position here, but supported tensor length does not establish useful page-level context or
acceptable memory/runtime. The mmBERT model family also includes a base encoder at width 768;
it can be a later comparison if small helps. Prefer screening a few candidates over a full grid.

Adaptation:

- Use the base encoder, replacing vocabulary lookup with the learned page-fusion projection to
  the checkpoint's hidden width. Replace the original task head with our three heads. Avoid
  instantiating the vocabulary prediction head. Bypass/remove unused vocabulary parameters when
  practical; otherwise freeze and exclude them from the optimizer. Report actual loaded, active
  and trainable parameter counts rather than equating checkpoint size with page-model size.
- BERT and ModernBERT expose inputs_embeds in their official Transformers APIs. Pin model and
  library revisions; check external-embedding scale, masks, padding and gradients with a small
  forward/backward smoke test. Never silently fall back to vocabulary IDs for pages.
- Preserve the checkpoint's position mechanism: BERT's learned positional embeddings or the
  modern encoder's rotary positions. Do not also add the scratch model's sinusoidal encoding.
  Start with supported local window lengths; use window-local positions consistently in training
  and inference. Do not feed arbitrary global scan offsets beyond learned position limits.
- Optionally warm up fusion and heads briefly with the encoder frozen, then unfreeze with a
  smaller encoder learning rate. Compare this with immediate unfreezing on development data;
  a frozen text encoder may never adapt sufficiently to page tokens.
- Compare pretrained versus random weights using the exact same encoder configuration, fusion,
  position mechanism and training budget. The four-layer scratch model remains an efficiency
  baseline, but its comparison with a 22-layer pretrained model does not isolate pretraining.
- English versus multilingual text pretraining is not a direct language-coverage guarantee once
  the input becomes page vectors; that coverage already enters through frozen page embeddings.
  Measure transfer, including the pretrained-minus-random difference after full dump adaptation.

[ModernBERT's API](https://huggingface.co/docs/transformers/en/model_doc/modernbert) and
[BERT's API](https://huggingface.co/docs/transformers/en/model_doc/bert) document external embedding
inputs. This permits the proposed integration; it does not validate its transfer performance.

[CoSMo's repository](https://github.com/mserra0/CoSMo-ComicsPSS) is a related page-stream reference,
but the checked material does not establish a ready checkpoint compatible with our stored vectors
and label space. Do not treat it as a drop-in initialization without checking its artifacts and
feature contract. Models requiring raw page pixels, OCR words or boxes would change the input
pipeline and are separate later experiments rather than replacements for the vector encoder.

## Constructing scan streams and windows

CSV row order is not a scanning stream. Build a document index keyed by (library, doc), sorting
pages by numeric order and auditing ties, duplicates and gaps. Do not silently resolve ambiguous
orders with a UUID sort and then call that order ground truth. Index feature availability and
annotation references without reading private source text or OCR.

Generate a synthetic stream by joining complete retained document sequences and then taking
random contiguous crops. Only the outer crop may truncate documents in the initial sampler.
Joining arbitrary interior fragments and labeling the join as a physical document start creates
bad start-page examples. Treat incomplete-source joins as a separately flagged condition.

Use three sources of windows, with the stage-dependent mixture specified below:

- Coverage sampler: shuffled document/block traversal with randomized crop offsets, carrying
  tails across steps so edge pages are not systematically dropped. Plan an explicit coverage
  pass over every eligible training row; random sampling with replacement alone is not a claim
  that the model saw the whole dump.
- Interesting-span sampler: choose a known non-NormalPage anchor or a reliable document-start
  anchor, then crop a window containing it. Initially allocate two thirds of these draws to type
  anchors and one third to start anchors. DB types supply weak anchors; known annotations override
  them when deciding whether a page is NormalPage or a rarer type. Do not require document starts
  to have a cover/title label, and do not use every ordinary left/right alternation as an event.
- Annotated-span sampler: choose a curated annotation as the anchor, then crop a containing
  window. Any annotated type is eligible, including NormalPage. An annotated NormalPage is useful
  especially where it corrects a noisy DB type. Score all other available labels in the window.

For targeted type anchors, use tempered class balancing: start with a per-anchor weight
min(sqrt(N/n_class), 10), where N and n_class count eligible anchors within that source's fit
partition. Without the cap this gives class mass proportional to sqrt(n_class), between natural
sampling and uniform class sampling. Apply it separately to interesting type anchors and curated
anchors. The start-anchor part uses its own document-boundary index. Do not also apply full
inverse-frequency type-loss weights in the primary experiment. Compare with uniform curated
anchor sampling to see whether tempering helps without repeatedly fitting tiny classes.

Jitter anchor positions throughout the window rather than always centering them. A document-start
anchor needs at least one preceding visible page for its starts_document loss; otherwise mask
that target or draw a different crop. For the first retained page of the first document in a
synthetic stream, do not invent a preceding document just to make its first-position target valid.
Prefer a containing stream with actual internal joins when targeting boundaries. Repeated draws
of a page are repeated exposure, not additional independent annotations.

Keep long spans of normal pages and hard within-document changes: background continuity is a
real prediction problem. Targets are not exclusive categories: a coverage window can contain rare
pages and annotations, and a targeted window usually contains many NormalPage pages. Log the
observed window composition independently of its requested sampling source. In particular, a
quota of interesting anchors is not a quota of non-NormalPage loss terms.

For a controlled boundary-training arm, aim for approximately one third of windows with zero,
one, and multiple internal boundaries. Preserve complete internal documents; if the corpus cannot
supply a requested stratum at a given length, record this instead of fabricating boundaries.
Compare against naturally cropped streams. Short documents/edge pages will be oversampled in
the targeted arm, so log actual per-page and per-document exposure.

### Same-volume issue boundaries

Make difficult joins a primary training condition, not a rare side effect of random concatenation.
In particular, issue A's ending followed by issue B's beginning from the same volume is a positive
boundary even when layout, language, publisher, paper and text topic are very similar. Build an
eligible sibling-issue index using library and parent-volume identity; distinguish a volume parent
from unrelated collection parents when reading hierarchy metadata. Never mix fit/dev/test documents
to complete an otherwise unavailable sibling pair.

Initial target composition of sampled joins, separate from the 60/25/15 window-source mixture:

| Join stratum | Proposed share of sampled joins |
|---|---:|
| Different issues of the same volume | 60% |
| Same periodical, different volumes | 20% |
| Unrestricted distinct documents, including books | 20% |

These are pilot targets subject to eligible-pair availability, not assumed corpus frequencies.
Include both neighboring issues and other issue pairs from a volume. Use order/date metadata to
construct chronological runs as well as shuffled issue runs; do not make the ability to recognize
one particular issue order a prerequisite. Preserve complete internal issue sequences in either
case. Broad coverage still traverses all eligible documents; compose sibling runs where possible
and use the targeted sampler to supply additional hard joins without claiming those repeats as
new coverage.

Apply join-stratum selection when building streams for any sampling source, not only in a
separate final boundary experiment. An annotation anchored in an issue can be surrounded by other
issues from its volume. Book annotations remain eligible even if no issue-family join is possible.
For explicitly targeted starts, oversample eligible same-volume pairs; jitter the positive page's
window position, including difficult positions near artificial seams during evaluation.

Record requested versus realized join strata, unique volumes/issues/pairs, repeat counts, and
the number of actual windows containing each boundary type. A majority of hard joins does not
guarantee enough hard-boundary windows if most crops are deep inside long documents. Adjust the
start-anchor quota after inspecting this exposure. Balance among volumes/pairs rather than letting
a few convenient sibling pairs supply nearly all hard examples. If a stratum is unavailable in a
split, report and redistribute its quota explicitly; do not silently substitute easy joins.

Use hierarchy metadata only to choose and label examples, never as encoder input or positional
resets. Do not join copies of the same page across a purported boundary. Include hard non-boundaries:
chapter changes, illustration inserts, blank pages, title-like pages and page-number resets inside
one issue or book. Hold out some joining recipes for robustness evaluation.

The dump has no genuine cross-document scan-batch order. Constructed joins have known targets,
while their prevalence/order is a simulation. Real scan batches can later measure distribution
transfer; they are not required to define the initial issue-level boundary labels.

## Training schedules and loss accounting

The primary proposal uses all available supervision from the beginning. Broad pretraining and
fine-tuning share the dataset, architecture, target definitions and annotation precedence; the
main stage change is how frequently the sampler revisits curated spans.

Initial mixture for fixed-length windows (proposed pilot values, not measured optima):

| Source | Broad pretraining | Annotation-focused fine-tuning |
|---|---:|---:|
| Coverage windows | 60% | 20% |
| Interesting windows: rare types / starts | 25% | 15% |
| Annotated windows | 15% | 65% |

This keeps naturally sampled dump context as the largest source initially, while reserving 40%
of draws for targeted exposure. Fine-tuning continues to see non-annotated documents. Percentages
refer to draw sources, not guarantees about all-normal windows or the token-label distribution.
Tune against actual composition and full-stream development results. For variable window lengths,
implement quotas in valid processed pages or explicitly report the resulting page-budget change.
Use stratified source quotas across updates/gradient accumulation, with independent resumable
sampler states, rather than three streams racing independently to fill a queue. Coverage keeps
its own cursor as mixture weights change. Save counters and RNG state on checkpoint/resume.

Compare three schedules on the same model, manifests and initial seeds:

| Schedule | Training |
|---|---|
| Annotated-context baseline | Annotation-anchored windows; curated type loss, plus available placement/boundary losses |
| Sequential ablation | Same mixed broad pretraining, then curated-context fine-tuning without weak type loss |
| Continued mixture (primary) | Broad 60/25/15 mixture, then annotation-focused 20/15/65 mixture, retaining weak supervision |

Broad training uses curated type labels wherever present, otherwise valid weak DB type labels,
plus placement and document-start objectives. This is supervised pretraining with partly noisy
labels, not a DB-only phase. An optional later self-supervised arm masks 15% of page tokens
(including all their modalities) and reconstructs frozen dense embeddings from context using a
cosine loss. Score only masked, present targets. Compare against a context-free reconstruction
control: predictable generic pages can make reconstruction look good without helping tagging.

In every source and both stages, resolve type targets by annotation first, DB second, unknown
otherwise. Suppress weak type loss on annotated pages, including when annotation and DB agree;
do not teach a conflicting weak target alongside its annotation. Use every valid curated label
visible in a window, not just the selected anchor. Placement/starts remain metadata-derived
supervision unless checked labels become available; type annotation does not upgrade their status.

Reduce microbatch sizes or accumulate gradients to hold the total processed-page budget constant
when comparing mixtures. Combine the shared encoder gradients once per update. If the same page
appears in multiple windows of an update, divide its corresponding target weights by occurrence
count rather than silently increasing that page's loss contribution. Distinct context views can
still provide distinct gradients, with the total per-page target weight controlled. Repeated
annotation exposure across updates is intentional.

Initial loss decomposition:

    L = L_annotated_type + 0.2 * L_weak_type
        + 0.5 * L_placement + 0.5 * L_starts_document

Each term is averaged independently over its valid targets (or the sum of its target weights),
not over all pages in the stream. An empty target set contributes zero. Annotation sparsity must
not dilute its loss by the full window length. Log each term's target count and gradient scale.
Keep these coefficients fixed across stages in the first primary run to isolate the change in
sampling. Sampling probabilities and loss coefficients are separate controls; changes to the latter
are later ablations. A gradual shift over a short transition is an alternative to a hard stage switch.

Initially use no additional type-class weighting with tempered anchor sampling. Monitor effective
NormalPage versus other-type target counts and loss/gradient contribution: a 128-page targeted
window can still have just one rare page and 127 background pages. If weak background overwhelms
rare evidence, compare a capped NormalPage weak-target budget per window, keeping all pages as
model inputs. This changes supervision density, not context. Include it as a named sampling/loss
ablation rather than quietly assuming that interesting-window sampling solves token imbalance.
Keep placement and start prevalence visible. Select document-start thresholds on representative
development streams; oversampling starts changes calibration.

Never use weak label agreement as the main fine-tuning selection metric: the objective is to
improve on DB mistakes. Report curated type performance where DB agrees versus disagrees.

Track unique dump-page coverage, total processed pages, annotation presentations per class,
optimizer steps, and wall time. At minimum, match total processed pages/updates for the initial
schedule comparison; also report the annotation exposures, which will differ. Repeat promising
arms with matched curated exposures to separate schedule effects from extra annotation training.

Also log source quotas versus realized draws; all-normal spans, rare-type-containing spans,
boundary-containing spans and annotation-containing spans (overlapping flags); annotation counts
per window; anchor offsets; positive/negative starts; per-class and per-document repetition;
unknown-label fractions; and targets dropped or downweighted by any background budget. Detect
train-document/copy-group overlap with held-out splits before any sampler is allowed to run.

Define a coverage epoch by completion of the coverage cursor, not by a fixed number of mixed
updates. Targeted draws are additional presentations; they do not advance that cursor. Decide
pretraining length on the pilot throughput and development curves, with at least one completed
coverage pass if the experiment claims to train on the whole eligible dump. Fine-tuning is
governed by curated exposure and development metrics, with early stopping and retained coverage.

## Context length and inference

Start with 128 page tokens. Compare 32 / 64 / 128 / 256 on the chosen model, holding total
processed pages approximately fixed, and record GPU peak memory, pages/second and I/O wait.
No hardware-specific batch size or runtime claim is made before benchmarking.

Dense attention scores grow with sequence length squared: doubling length roughly quadruples
that part of the work for a fixed number of windows. For a fixed page budget, there are fewer
windows and the attention contribution grows roughly linearly with length. Other costs also matter.

The API accepts an arbitrary-length scan batch. Internally, initially use windows of 128 with
stride 64 and center-weighted logit merging. Windowing, merging, calibration and final decoding
form part of the evaluated model, with explicit settings saved alongside the checkpoint.

Randomize training crop origins and include variable lengths. Padding is not a blank page.
Some evidence farther than a window can be missed; promote 256 or a longer-context method only
if the measured quality/throughput tradeoff justifies it.

## Long-stream inference and overlap experiments

The evaluation unit is a complete scan stream, not an independently scored model window. Given
N pages, produce exactly N type/placement outputs and N page-aligned starts_document entries,
of which N-1 have visible adjacent-pair evidence. The initial start entry remains unknown unless
outside context is supplied; an output segment begins there by convention.
Give every page occurrence a stream position; a UUID alone is insufficient when pages recur.
Do not discard an initial page's type/placement just because its preceding boundary is unknown.

Compare the following with the same frozen checkpoint and stream manifests first:

| Strategy | Window size 128 example | Purpose |
|---|---|---|
| Minimal-overlap tiling | Stride 127; one shared page | Cheap control with coverage of every adjacent boundary pair |
| Keep central predictions | Stride 64; retain the central 64 pages per interior window | Remove predictions with little two-sided context |
| Uniform probability mean | Stride 64; average softmax/sigmoid outputs | Simple mixture retaining uncertainty |
| Uniform logit mean | Stride 64; average logits, then softmax/sigmoid | Test a different consensus rule |
| Center-weighted mean | Stride 64; weighted logits initially, probabilities as an ablation | Prefer predictions farther from artificial window edges |

Strictly nonoverlapping windows leave an adjacent-pair boundary head without a prediction across
each seam. If included as an extra diagnostic, provide a separate seam pair pass and charge its
cost; never fill those gaps with a fixed positive/negative answer. For central selection, assign
each boundary pair separately to a window that contains both pages, preferably centrally; taking
page i and page i-1 from different windows does not create a valid boundary-head prediction.

For each page i, window k gives logits z(k,i) and a nonnegative weight w(k,i):

    merged_logits(i) = sum_k w(k,i) * z(k,i) / sum_k w(k,i)

For probability merging, substitute probabilities and omit the final softmax/sigmoid. Compare
these rules rather than assuming equivalence: normalized logit averaging pools log odds and can
behave differently from averaging probabilities. Do not sum logits as if overlapping predictions
were independent evidence, or choose the window with maximum confidence by default.

A reproducible initial center weight for real page position r in a window of real length l is:

    w(r) = 0.1 + 0.9 * min(1, (min(r, l-1-r) + 1) / 32)

Use the minimum weight of the two pages for boundary pairs. Missing pairs contribute neither a
prediction nor a denominator. The positive floor handles outer edges; choose normalization and
weighting constants on development streams. Do not downweight genuine scan starts/ends as though
another window could supply unavailable outside context: for a window touching a real stream
endpoint, omit that side from the minimum-distance calculation (both sides omitted means weight
one). For central selection, explicitly extend the first/last kept regions to cover those stream
endpoints.

Create deterministic, ordered start positions; cover the stream tail without duplicating its last
window. Handle streams shorter than one window and the last partial window, using masks and real
lengths. Assert positive merge weight for every page and every internal pair. Accumulate sums in
float32, excluding padding; finalize outputs only after all contributing windows have been seen.
Keep memory bounded by active windows plus output accumulators, not every hidden state.

Merge separately for each task, then apply development-fitted calibration/thresholds, then decode
segments once over the entire stream. If a CRF is enabled, use merged unary scores and a global
decoding pass; do not restart it at every window. Display segment one from the scan's first page
by convention without counting a forced initial boundary as a model success.

For window length W, compare stride W-1, W/2, and W/4 where feasible. The nominal processed-page
multiplier is about W/stride for long streams, with endpoint overhead measured exactly. Compare
quality versus measured unique output pages/second and latency, including merging and decoding;
do not report only the speed of individual windows. Start with one checkpoint to screen merge
rules, then recheck shortlisted rules across model sizes and context lengths.

Additional diagnostics:

- Shift the tiling origin while keeping the input stream fixed. Supply prefix/suffix windows so
  coverage remains complete. Report prediction disagreement and worst/mean quality across offsets.
- Bin errors by distance to a window edge and by true-boundary position modulo stride. Include
  boundaries directly at seams, plus long documents with many seams and no true boundary.
- Measure disagreement between overlapping probabilities, calibration and final errors; retained
  overlap counts are a diagnostic, never an implicit confidence multiplier.
- A second pass shifted by half a stride is a later compute/quality arm. An uncertainty-triggered
  rerun with a better-centered window is another, with its trigger tuned on development and total
  runtime charged. Neither can recover arbitrarily distant context beyond the model window.
- On short supported streams compare one full-window pass with chunked inference. On long streams,
  compare against a larger-context checkpoint if available. These quantify chunking loss; neither
  reference is ground truth, and overlapping inference is not equivalent to global attention.

If all overlap methods miss dependencies farther than W pages, test a second, small sequence
encoder over coarse page-block summaries, or a recurrent memory model, using the same stream
evaluation. Do not claim that choosing a better merge rule alone solves missing long-range context.

## Evaluation

- Type: existing weighted F1 as the primary reference, accuracy, macro F1, per-class
  precision/recall/F1/support, and document-group bootstrap intervals for shortlisted comparisons.
- Placement: macro F1, per-class recall and confusion matrix on known targets; separate a manually
  checked subset from DB-agreement scores. Check long runs and pages next to boundaries.
- Boundaries: exact precision/recall/F1, PR-AUC, false splits per 1,000 internal non-boundary pairs,
  missed-boundary rate, and exact recovery of whole document spans. Report tolerance +/-1 only
  as an additional diagnostic. Exclude unknown initial-window pairs and explain censored spans.
- Joint behavior: type/placement accuracy near versus far from boundaries, page-level joint
  correctness where every target is checked, and numbers of over-split and merged documents.
- Deployment behavior: actual chunked path, missing modalities, short/long documents, same-library
  and same-parent joins, unseen joining recipes, incomplete documents and order gaps.

Use fixed streams at several document-length/boundary-density settings. Synthetic stream
resampling must not silently change the weight of curated type test pages: evaluate each curated
page once in a canonical stream, and report repeated-placement stress tests separately. Report
boundary/placement DB agreement separately from manually verified performance.

For starts, always report same-volume issue, same-periodical cross-volume, and unrestricted
join metrics separately, plus their sample counts and within-issue false-split rates. Maintain a
fixed hard-join development/test suite with sufficiently many distinct volumes and issue pairs;
bootstrap/group results accordingly when sibling issues are correlated. Also report results on
the broader mixture. An aggregate dominated by easy book joins cannot establish success on
same-volume issue segmentation. If comparing operating points, use a shared development-chosen
threshold and show per-stratum precision/recall rather than choosing thresholds on test strata.

Construct stream-length bands around 512, 2048 and 8192 pages initially, in addition to short
streams; extend these when actual scan-batch sizes are known. Preserve internal complete documents
and allow target-length variation rather than forcing a cut at every band limit. Include documents
longer than several model windows, multi-document streams, very short documents and similar
neighboring documents. Report the full-stream distribution, not just nominal requested lengths.

All pipeline variants see the identical page order, endpoints and labels. Merge every window
prediction first, then score each page/pair exactly once. Report pooled page/pair metrics plus
per-stream scores by length band; never average window F1 values. Measure exact document-span
recovery and the fraction of completely correctly segmented streams, since small local error rates
can still yield poor long-stream recovery. Exclude or separately report censored outer spans.
When resampling multi-document results, resample whole streams; for type-only document estimates,
resample source document/copy groups. Repeated tilings/stream orders are paired robustness runs,
not independent additional test documents.

Select checkpoint and inference pipeline on complete development streams, with fixed manifests
for routine validation and separate stress manifests for offsets and joining recipes. Avoid picking
checkpoints solely on center-cropped windows and then assuming they rank the same after merging.
Choose final settings and thresholds before the final test; retain a small benchmark table of
window length, stride, merge rule, type/placement scores, boundary errors, exact spans, throughput,
peak memory and processed-page multiplier. This is the main deployment comparison.

Log all three task metrics. Initially shortlist checkpoints by curated type development weighted
F1, but retain placement/boundary tradeoffs rather than hiding regressions in one aggregate score.
Thresholds and final operating points require development data representing the intended workflow.

## First-run recipe and remaining decisions

The plan above contains several later ablations. Keep the first comparison small:

| Run | Encoder initialization | Fusion |
|---|---|---|
| P0 | Page-wise MLP and adjacent-page start baseline | Joint preprocessing/fusion |
| P1 | Pretrained BERT-Mini | Single joint linear projection |
| P2 | Random BERT-Mini, exact same configuration as P1 | Same as P1 |
| P3 | Pretrained BERT-Mini | Separate nonlinear modality branches, then linear fusion |

Use all grouped normalized statistics, mmBERT page embeddings and DINOv2 in every arm. Use
128-page windows, the same manifests and sampling schedule, and one fixed seed for screening.
P1/P2 isolates pretrained initialization. P1/P3 tests a complete fusion design change, including
capacity; if it helps, add a parameter-budget-matched joint MLP to identify whether separate
branches help beyond additional nonlinear capacity. CNN and larger pretrained encoders can follow.
Repeat close shortlisted comparisons with three seeds before claiming a reliable improvement.

Proposed optimizer defaults, to validate on the pilot:

- AdamW; encoder learning rate 3e-5; newly initialized fusion/heads 3e-4; weight decay 0.01,
  excluding biases and normalization parameters; gradient-norm clipping at 1.0.
- Linear warmup over the first 5% of the pilot updates, then constant learning rates for the
  initial sampling comparison. Retain optimizer state and learning rates when changing mixtures
  so that the sampler change can be assessed. Lower-rate fine-tuning is a subsequent ablation.
- Start with float32 on the RTX 3090 for correctness checks; benchmark bfloat16/autocast against
  float32 for finite losses, gradient behavior and actual throughput before adopting it.
  Normalize stored float16 vectors in float32. No learning-rate or tensor-dtype choice is claimed
  optimal before this check. Use the same effective page budget across compared models.
- Benchmark microbatch size and loader workers on this GPU/SSD, then fix gradient accumulation,
  processed-page budgets for both pilot stages and a full-stream validation interval. Record
  these values in run configs before launching comparisons, not after seeing results. Checkpoint
  at validation and retain a resumable latest state, including optimizer/sampler/RNG counters.

Data-dependent questions to resolve at the audit/benchmark step:

1. Input assumptions: retain ordered single-page inputs. The document target is settled at the
   DB leaf-document level (issues within volumes). Audit the availability of distinct same-volume
   issue pairs in each split and choose reproducible join manifests with explicit hard-join counts.
2. Supervision validity: count placement coverage/values, order ambiguities, duplicate/copy groups,
   retained-endpoint completeness, feature availability and invalid statistics. Establish explicit
   masks and repair/quarantine counts. Distinguish DB-derived labels from checked evaluation labels.
3. Pilot budgets: fixed comparable processed-page budgets plus annotation-exposure counts. Ensure
   the fine-tuning pilot has useful coverage of curated classes rather than only repeated easy
   anchors. A pilot need not cover the full dump; the later full run must finish its coverage cursor.
4. Selection/stopping: select by curated type weighted F1 on fixed full development streams,
   reporting placement and start tradeoffs for each checkpoint. Set early-stopping patience and
   maximum budgets after measuring validation cost, before model comparison. Tune start thresholds
   separately on representative development streams; do not use a fixed 0.5 as an assumed optimum.
5. Inspection artifacts: retain keyed page-level predictions, confidence distributions and masks,
   plus a small public-data error report showing predicted versus target labels around starts and
   annotation/DB disagreements. Source-private runs produce vectors/metadata/predictions only.

Before long training, require: tiny-example overfitting of all three heads; finite-loss/backward
checks; masked-label/padding invariance; no attention across separate training examples; exact
prediction alignment through overlap merging; complete page/pair coverage; deterministic sampler
resume; and timing that separates vector I/O from GPU work. Compare against NormalPage/majority-side
and no-start trivial baselines so a superficially good aggregate cannot hide a failed head.

## Run order and preparation before SSD arrival

1. Specify the manifest schema, masks and output contract with leaf-document/issue boundaries.
   Build same-volume join indexes and evaluation strata. Plan the placement-label audit.
2. Implement manifest building and split audits with fixture data; normalization/fusion; coverage
   and annotation samplers; three heads; masked losses; and overlapping inference. Test meaningful
   invariants: no split overlap, no chunk-edge boundary artifacts, no leaked IDs, correct handling
   of missing modalities, and correct independent loss denominators.
3. CPU smoke test and deliberately overfit a tiny synthetic example for all three heads. Establish
   deterministic resume/checkpoint behavior. These checks do not require private source material.
4. On SSD arrival, audit metadata and vector coverage, build document indexes, fit train-only
   transforms, and benchmark lazy LMDB reads. Open stores per worker; avoid loading 45M rows or
   Python dictionaries of all vectors into RAM. Cache/repack into document-contiguous shards only
   if I/O measurements justify the extra disk space and preparation cost.
5. Run page/pair MLP, CNN and Transformer on the same small training subset, context 128, with the
   same fusion and mixed objectives. Add pretrained BERT-Mini versus a random model of exactly
   its configuration, then mmBERT-small versus its random control if resources allow. This is a
   screening comparison, not a full-corpus result. Screen overlap rules on full development
   streams with one fixed checkpoint, then recheck shortlisted pipelines on the selected models.
6. With the selected encoder, compare the three training schedules on a fixed pilot budget;
   compare 64/128 first if resources are tight. Expand to 32/256 and three seeds for close winners.
7. Run the selected coverage-plus-mixture schedule across the full eligible training dump, with
   periodic curated development evaluation. Then test modality/fusion ablations one at a time:
   statistics removal, Qwen3 substitution, both text models, and full statistics versus grouped.
8. Add CRF, masked reconstruction, or encoder-decoder experiments only when a specific remaining
   error pattern motivates them. Reserve the final test comparison for selected configurations.

Save configs, code revision, data/vector versions, split/stream manifests, normalization state,
seeds, processed-page counters and checkpoints. Logs contain IDs, counts, predictions and metrics,
never private source text, OCR, snippets or vocabularies. Follow CLAUDE.md storage restrictions;
temporary data and run artifacts go to an explicitly configured directory, initially under
/home/ikohut/tmp/. No access to /mnt/zfs1 is part of this plan.

## Research context

- [Wiedemann and Heyer, 2018](https://aclanthology.org/L18-1581/) studies text/image page stream
  segmentation. It establishes the closely related task; its reported results are not a baseline
  measurement for this corpus.
- [CoSMo, 2025](https://arxiv.org/abs/2507.10053) studies multimodal Transformer page stream
  segmentation for comics, a different domain. It motivates a sequence-model comparison without
  establishing which architecture will work best here.
- [Huang et al., 2015](https://arxiv.org/abs/1508.01991) compares recurrent sequence taggers with
  CRF decoding. Applying structured decoding to page labels is an experiment proposed here.
- [BERT, 2019](https://arxiv.org/abs/1810.04805) provides the bidirectional masked-prediction
  precedent. Reconstructing frozen page embeddings is an adaptation, not the original objective.

## First corpus launch settings (2026-10-25)

The agreed initial run uses pretrained BERT Mini, separate nonlinear modality projections,
FP32 throughout, batch size 64, 128-page windows, and eight loading workers (prefetch factor 1).
These choices supersede the earlier joint-linear/synchronous pilot suggestions above.
ClearML records configuration, metrics, console and resource usage; model/checkpoint files stay local.
The full-corpus metadata preparation is streamed into SQLite, with approximately 15% development
document groups and the existing test partition excluded from both training stages.
Initial launch is gated on the completed private statistics LMDB path and data preflight.
