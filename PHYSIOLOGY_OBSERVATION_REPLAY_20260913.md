# Persistent Observation Replay Contract

The state layer now has an exercised chronological prediction/correction path:
`advance` performs bounded time integration, `update` corrects an already
predicted state without advancing its clock, and `replay_observations` applies
both to grouped strictly increasing available-time observations.

Previously both transition and update incremented elapsed_hours. A caller
composing them with the same interval would count that interval twice. No
inspected runtime caller exercised that composition, so this was an integration
contract defect, not evidence of double-counting in prior MAP training.

Replay keeps one body and all organ states, corrects only present organs, and
allows physiological transition between events. Absent observations can contain
NaN and produce prediction-only behavior. Duplicate timestamps are rejected:
simultaneous observations must be grouped upstream, not silently consumed twice.
Event identity deduplication, availability extraction and semantic organ encoding
remain caller responsibilities; this method does not certify raw-data validity.

25 tests pass across replay, missingness, physiology state and organ exchange.
New tests cover a2.5h prediction/update interval counted once, a two-organ
alternating observation sequence identical to manual integration, gradients to
both observed organ inputs, no gradient to missing organ input, prediction-only
missing events, and rejection of duplicate/backward timestamps.

This is a tested latent-stream component, not a trained clinical replay model.
It has not yet been connected to the availability-native JEPA input pipeline,
nor shown future MAE or calibrated-interval benefit. No weights, registry or
calibrators changed. The existing future-rollout path was not modified.
Prior update API source bindings remain historical, not validation of this
changed time contract. No promotion.
