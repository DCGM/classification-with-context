# NDK-informed output consistency and ordering augmentation

Research snapshot: 2026-10-27. This is a proposal and a read-only metadata audit;
it does not change the active r15 experiments or implement a new loss.

## Sources and what they establish

The current official description rules are [monographs 2.5](https://old.ndk.cz/standardy-digitalizace/ppp_mono_2.5_final.pdf)
and [periodicals 8.8](https://old.ndk.cz/standardy-digitalizace/ppp_perio_8.8_final.pdf),
both dated 2026-10-21. The live official index and actual PDFs were checked;
the browsing service's cached index still listed the preceding versions.
Public copies and extracted text are under
`/home/ikohut/tmp/ndk-consistency-study-20261027/`, with PDF hashes recorded in
the [aggregate audit](ndk-consistency-metadata-audit.json).

Particularly important findings:

- FrontJacket means the representative jacket image placed first; it is not
  synonymous with Jacket. TitlePage can describe both pages of a spread or two
  title pages. Colophon can occur at either end of a book and at various locations
  in a periodical. Frontispiece denotes a left-hand facing title illustration.
  FlyLeaf denotes a loose insert, not necessarily an English-language front
  endpaper. The new GeneralView type is outside our current 39-class vocabulary.
  See [monographs 2.5, §§1.2.1–1.2.2](https://old.ndk.cz/standardy-digitalizace/ppp_mono_2.5_final.pdf#page=8).
- Digital ordering of binding assets has alternatives, including assets after
  the book block. BackCover need not be the final scan: spine, jacket and loose
  inserts may follow. Consecutive blank pages are explicitly discussed.
  See [monographs 2.4, §§1.2.1 and 1.3](https://old.ndk.cz/standardy-digitalizace/ppp_mono_2.4_final.pdf#page=15),
  retained in version 2.5.
- Periodicals can contain dedicated issue-level groups of covers; annual
  contents/index/cover material can also form separate units. An issue therefore
  need not contain an ordinary body or have a unique cover/title sequence.
  See [periodicals 8.7, §§3.6 and 5.4](https://old.ndk.cz/standardy-digitalizace/ppp_perio_8.7_final.pdf#page=26)
  and the updated supplement treatment in [8.8 §5.4](https://old.ndk.cz/standardy-digitalizace/ppp_perio_8.8_final.pdf#page=35).

NDK describes digital representation as well as the physical source. A physical
printing convention is not automatically a constraint on scan order. In addition,
a complete scan of the available archival object can lack original printed covers
or leaves. Neither implies that scanning stopped prematurely.

For comparison, [Chicago's official pagination Q&A](https://www.chicagomanualofstyle.org/qanda/data/faq/topics/ManuscriptPreparation/faq0033.html)
distinguishes the strong odd-page/right-side convention from the weaker convention
that a contents section begins on the right. The authors' own
[Book Design Made Simple excerpt, pp.24–25](https://www.bookdesignmadesimple.com/wp-content/uploads/2015/08/Book-Design-Made-Simple-2ndEd-Part1.pdf)
illustrates optional sections, a two-page contents spread, and material that may
occur in different locations. Such modern left-to-right book conventions are
useful priors, not universal rules for historical and multilingual collections.

## Read-only fit metadata audit

Sample: 1,492 fit documents / 135,829 pages. Selection combined a deterministic
document-ID sample (1,008 documents) with an annotation-enriched stratum (486,
two overlapping). No dev/test rows, OCR, images or embeddings were inspected.
The audit streamed per-document metadata through a read-only SQLite connection
with a 4 MiB cache and disabled mmap; it took 4.74 seconds. It is not a
corpus-unbiased prevalence estimate.

There were 920 human-labelled page types and only 119 adjacent pairs with both
types human-labelled. Placement/side values are DB metadata even when the page
type is human-confirmed. Adjacency counts omit 26 observed order gaps.

| Observation | Human page-type annotations | Weak DB page types |
|---|---:|---:|
| FrontJacket at actual document first position | 55 / 55 | 109 / 111 |
| Adjacent TitlePage → TitlePage | 1 pair | 14 pairs |
| BackCover at actual document last position | 63 / 66 | 272 / 545 |
| Adjacent Blank → Blank | No observed labelled pair | 256 pairs |
| Adjacent TableOfContents → same | No observed labelled pair | 494 pairs |
| Adjacent Index → same | No observed labelled pair | 1,484 pairs |

FrontEndSheet has 151 left, 4 single and no right DB placements in this sample.
BackEndSheet has 145 right, 4 left and 3 single. For human-labelled BackEndSheet,
42 have a right placement and 23 lack placement. These are useful leads, but
do not make the side metadata human ground truth. Frontispiece's sampled pages
have no placement labels, so this sample cannot validate that type/side rule.

The strongest empirical lead is FrontJacket implying document start. However,
55 human examples are limited evidence; zero observed exceptions is not proof
of impossibility. DB-only repetition and transition counts require review rather
than automatic acceptance as legitimate or rejection as errors.

Full aggregates: [ndk-consistency-metadata-audit.json](ndk-consistency-metadata-audit.json).
Reproduction script: `/home/ikohut/tmp/ndk-consistency-study-20261027/audit.py`.
Its SHA256 and the published catalogue identity are recorded in the aggregates.

## Candidate consistency terms

These are proposed soft losses, conditional on eligible representations and
reliable source definitions. They are not unconditional decoding restrictions.

| Candidate | Scope and exclusions | Assessment |
|---|---|---|
| FrontJacket implies document-start auxiliary output | Actual representative-image semantics; include true first scan page, whose pair-boundary output is intentionally invalid | Best first new type/boundary relation to audit more broadly |
| Frontispiece agrees with left-side output | Individual physical pages; merged/composite representations and unknown conventions need separate handling | Strong definition, insufficient local side evidence |
| FrontEndSheet favours left; BackEndSheet favours right | Conditional acquisition layout; allow combined/singlePage cases; investigate conflicting DB placements | Promising type/side soft prior |
| Left/right alternates across an ordinary page run | Same document, reliable physical continuity, no gap/composite/insert/binding jump | Useful but unsafe as a universal transition rule |
| Printed-number parity agrees with side | Conventional left-to-right page numbering; exclude foliation, spreads and unreliable/inferred numbers | Our current unbracketed DB candidates are noisy, so eligibility is critical |
| Binding/front/body/back phase ordering | Complete document and known scan recipe, including alternative asset ordering | Later structured model or soft grammar; not one fixed class order |
| At most one representative FrontJacket per document | Complete correctly scoped document; not an entire multi-document scan | Candidate diagnostic first; not a generic uniqueness penalty for every class |

Existing boundary/start/end consistency already couples the pair-boundary head
with start(current) and end(previous). That is a separate tested loss option;
it should not be bundled with new type-based rules in the first ablation.

Rules to avoid: forbidding every repeated type; forbidding consecutive title
pages; requiring a NormalPage, TitlePage or cover in every document; requiring
BackCover to be last; forcing Colophon to the back; making all covers right/left;
requiring exactly one title page; or forbidding a page from being both a document
start and end. A one-page document legitimately has both boundary properties.

## Realistic alternative-order augmentation

Alternative valid acquisition orders should also be represented in training.
This is particularly valuable for detached binding/cover assets: it teaches
robustness rather than punishing a legitimate ordering.

Construct a complete document, select one allowed acquisition recipe for its
verified asset roles, then construct a multi-document scan and tile it. Move
coherent asset blocks, preserving their internal order and the reading order of
the content block. Do not independently shuffle pages based only on a weak DB
label. FrontJacket retains its representative-first role.

The fact that a section can be published in different locations does not by
itself make every existing page a safe relocation candidate. Printed pagination,
left/right orientation, OCR content and cross-references remain attached to the
same image. A proposed move must leave a plausible sequence. Detached unnumbered
binding assets are therefore a cleaner initial augmentation than moving numbered
sections such as an existing Colophon across the text block.

Implementation requirements for a future augmentation experiment:

1. Separate immutable storage row IDs from the synthetic document order. Read
   vectors by their original IDs; present pages in the selected acquisition order.
2. Derive document-start, document-end and pair-boundary targets from the new
   order. Current native auxiliary targets use catalogue first/last IDs and must
   be overridden; retaining them would silently teach wrong targets.
3. Derive scan-first/last flags after the reorder. Keep metadata-provided and both
   document input bits zero for the scan-only experiment. If a later experiment
   supplies segmentation, update it completely and coherently for the whole scan.
4. Keep type, side and printed pagination attached to their source pages. Do not
   fabricate page numbers or recompute side merely to alternate the new sequence.
5. Mark synthetic binding joins separately from physically adjacent content
   pages. Do not apply pagination/side-alternation losses across these moves.
6. Checkpoint the recipe choice, ordering and pending scan tiles. Preserve coverage,
   annotation anchors, copy exclusions and fit/dev/test assignments.

Evaluate original and allowed alternative orders on the same pages, reporting
type accuracy/F1, ending-region performance, boundary performance and prediction
agreement across orders. A valid alternate input need not produce identical
confidence, but should retain the correct page identities and task labels.

## Loss mechanics and experiment order

A simple proposed implication penalty is:

```text
mean(P(type = FrontJacket) * (1 - P(auxiliary document start)))
```

This is a bounded differentiable surrogate, not a calibrated probability of
violation or the exact semantic loss from the literature. Both heads receive
gradients; ordinary supervised losses remain necessary to avoid satisfying a
rule by abandoning the correct class. The general idea of constraining neural
outputs has a formal precedent in [Xu et al., ICML 2018](https://proceedings.mlr.press/v80/xu18h.html).

For transition penalties, use a small weighted table of independently justified
unlikely pairs. Multiply only eligible neighbouring type probabilities, normalize
by eligible support and exclude known inter-document joins. Initially use actual
training segmentation for eligibility: gating by a trainable predicted boundary
would let the model reduce the penalty by inventing document splits.

Never equate a window endpoint with a document endpoint. Count/position rules
need a complete document or known true edges. Gaps and combined scans require
their own eligibility masks. Never turn an unobserved rare transition into a
hard impossibility or estimate the rule table from held-out test labels.

Suggested sequence after the current baselines:

1. Add violation diagnostics on selected checkpoints; review small sets of
   high-confidence contradictions and confirm label definitions.
2. Test verified alternative-order augmentation as its own ablation.
3. Test one high-confidence type/boundary relation with a small separately logged
   weight, then type/side constraints if their acquisition eligibility is known.
4. Consider a mixture of allowed scan grammars only if simpler terms help.

For each rule report eligible support, human/DB label contradictions separately,
predicted violation mass/rate and the original task metrics. Lower violation
counts alone are not evidence of improved classification or segmentation.
