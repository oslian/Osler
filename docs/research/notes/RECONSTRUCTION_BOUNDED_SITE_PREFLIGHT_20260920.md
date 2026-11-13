# Bounded Site Probability Preflight

Implemented inverse-site probabilities projected to relative weights [.5,2]
with mean one. Only source-fit patient/site membership is used; no labels,
predictions, encoder-selection results or evaluation outcomes are consumed.
Five tests verify hard bounds, normalization, equal/single-site behavior,
permutation equivariance, invalid inputs and deterministic RNG replay.

Across available source target pools:
- Selected group median site-mass total variation: .139130.
- Broad group median site-mass total variation: .275616.
- Minimum effective patient fraction: .671771 selected, .676501 broad.

These are distribution properties, not accuracy or calibrated uncertainty.
The rule changes site exposure meaningfully while avoiding the earlier49-80x
expected individual weights. It still reduces effective patient count and
with-replacement sampling can repeat individuals beyond their expected count.
Consequently a matched uniform with-replacement control is essential; old
without-replacement checkpoints are not sufficient to identify the effect.

The repair is implemented at probability/sampler level but not yet trained.
All33trained targets and six pending targets retain their existing status.
No runtime, checkpoint, registry, interval or promotion changes.

Evidence: artifacts/native_reconstruction_bounded_site_sampling_20260920.json.
Protocol: RECONSTRUCTION_BOUNDED_SITE_PROTOCOL_20260920.md.
