# Whole-document exposure with the current 512-page sampler

Estimate made on 2026-10-27 for the NVMe r12 configuration. This analysis does
not change sampling or the running experiments. Length means retained catalogue
page records, as in the [document-length statistics](../README.md#bottom-level-document-lengths).

## What the sampler actually does

[`WindowSampler`](../page_sequence/sampling.py) uses a maximum of 512 pages,
not 512 on every draw: 60% request 512, 20% request 32–511, 10% request 8–31,
5% request 2–7, and 5% request one page. Mean requested length is 363.725.

Coverage walks shuffled document/volume runs and cuts when the requested page
budget is full. It continues at the saved offset on the next coverage draw.
Anchored draws place a selected page at a uniform random window offset, then
fill the surrounding space with compatible documents. Interesting draws use
label anchors two thirds of the time and explicit boundary anchors one third.
Pretraining source quotas are 60/25/15 coverage/interesting/annotated;
fine-tuning quotas are 20/15/65. After sampling, non-coverage draws can be
cropped to a document start (10%) or end (10%), subject to retaining annotated
pages. Those crops can remove some documents from a window.

## Guaranteed exposure from existing counters

Two or more internal boundaries guarantee at least one complete intervening
document. This is a lower bound: windows with zero or one internal boundary
can also contain a complete document when they align with its outside edge.

| Phase | Counted windows | Guaranteed complete-document windows | Lower bound |
|---|---:|---:|---:|
| Pretraining, through step 15,427 | 987,520 | 695,635 | 70.44% |
| Fine-tuning after that snapshot, through step 17,067 | 105,472 | 79,889 | 75.74% |

These are sampler draw counts, including prefetched windows, derived from the
NVMe run's cumulative `source:*` and `windows_boundaries:multiple` counters in
`metrics.jsonl`. The fine-tuning row subtracts the final pretraining snapshot.
They include all requested lengths and applied crops. Existing counters do not
cross-tabulate completeness with window length, so they cannot give an exact
512-only percentage or identify which annotated document was complete.

## Probability of retaining the selected document

For an anchored document of length L and a fully filled window of length W,
the uniform anchor offset has `max(0, W-L+1)` acceptable positions out of W.
Thus, before edge crops, its full-document probability is
`max(0, W-L+1) / W`. This assumes enough compatible neighbours are available;
failed neighbour selection and clamping at a scan edge can change it.

| Document pages | In a requested 512-page window | Averaged over requested lengths |
|---|---:|---:|
| 128 | 75.20% | 53.77% |
| 220 | 57.23% | 38.78% |
| 256 | 50.20% | 33.41% |
| 384 | 25.20% | 15.85% |
| 492 | 4.10% | 2.48% |
| 512 | 0.20% | 0.12% |

A metadata-only sample of 3,000 anchors per source, using the sampler's actual
class masses `count * min(sqrt(total/count), 10)` and uniform slots, gives:

| Label-anchor source | Selected document complete, requested 512 | Across requested lengths |
|---|---:|---:|
| Interesting | 64.31% | 49.19% |
| Annotated | 63.00% | 45.97% |

These are formula-based estimates before edge crops, not complete sampler
replays. The 512 estimates have sampling standard errors of about 0.60 and
0.52 percentage points, respectively (NumPy generator seed 20261027; interesting
sample drawn first). They exclude the explicit-boundary branch and do not
count complete neighbouring documents.

For coverage, a stationary random-cut approximation with the fit length
histogram suggests about 92.52% of documents survive uncut with fixed 512
windows, or 91.66% with variable lengths. Those complete documents account for
about 53.22% / 51.01% of coverage page occurrences. These are per-document/pass
estimates, not per-window probabilities; volume-run correlations, compatibility
breaks, and pass endpoints are omitted.

## Sampling implication

Whole short documents already occur frequently. A general increase in
complete-document windows is not clearly needed. The gap is documents near
the context limit: a 384–512-page document often gets cut even when it fits.
A focused future comparison could bias 10–20% of eligible label-anchored draws
to an offset that retains the entire selected document, while preserving the
source quotas, annotated anchor, shorter scans, and isolated-edge crops.
Remaining capacity can still contain neighbouring documents; where possible,
retain a preceding page so the first document boundary also has pair-head
supervision. This is a proposal, not an implemented sampler change.
