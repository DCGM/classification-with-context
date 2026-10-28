# MetaKat boundary-feature inference parity

The selected W512 r16 fine-tuning checkpoint (weak DB weight 0.05, step 15927)
was exported to `metakat_context_v4`. Source SHA256:
`e52a49a453ce844e449d92918cf359260192e13bf9a7edb34b681cbff0ac5c4a`.

Deployment uses window 512, overlap 64 (internal stride 448), regular short
tails and centre-weighted cubic logits. The grouped `overlap_merge` policy
sets `ramp_fraction_of_overlap=0.75`, `ignore_fraction_of_ramp=0.25` and
`power=3`, resolving to ramp 48 and ignore 12. There is no additive weight
floor. Changing overlap rescales both distances. The five-bit boundary
MLP, pair boundary head and auxiliary start/end heads load strictly. Scan
endpoints are inputs; complete document metadata is optional. CUDA uses FP16
autocast with FP32 weights and merge accumulators.

`metakat_checks.compare_context_inference` compared the exported real model in
the classification environment (Python 3.14) with the MetaKat engine loaded in
its own environment (Python 3.12), on CPU. Inputs had 1, 17, 512 and 1024 pages,
each with unknown or complete document metadata. Fixed synthetic features
included missing modalities; each reference window ran separately while MetaKat
batched windows with padding. The 1024-page case ended with a 128-page window.

All eight cases passed. Type, side and boundary decisions matched exactly,
including pair masks and segment IDs. Page/pair merge weights matched exactly.
All five heads' merged probabilities agreed within 0.000002 absolute error.
The real artifact also passed the engine's feature/label/normalization contract
validation; image/OCR feature encoders were mocked. This does not establish
feature extraction parity on archived images and ALTO, nor compare the separate
boundary resolver's final decisions with raw model predictions.

Local artifact and detailed results:
`/home/ikohut/tmp/metakat-boundary5-r16/context.pt` and
`/home/ikohut/tmp/metakat-boundary5-r16/parity-report.json`.
