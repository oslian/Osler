# Native Organ Future Supervision Integration

Added a candidate state-change readout over the replayed anatomical trajectory.
All requested targets read their direct owner's state change on one fixed-hour
future integration grid, plus target anchor/mask and query time. Cross-organ
edges and pooled-body feedback remain closed. Value and target-latent residuals
start exactly zero over supplied frozen JEPA baseline predictions. Value residuals
are not artificially constrained to only shrink the baseline's delta.

One genuine source-fit MAP pair exercised two optimizer steps with the existing
value SmoothL1 and EMA target-delta JEPA loss. Nominal6h query and native paired
label were retained for this integration check; this is not new label-policy
validation or an actual-time efficacy comparison.

Step0 value loss0.983613, latent loss0.00513885, organ-update gradient0 (expected
with zero readout initialization). Step1 value loss0.982663, latent0.00513085,
organ-update gradient0.0000512096. This establishes real future supervision can
reach correction parameters after the first head update. It does not establish
useful learning, meaningful improvement, convergence, calibration or promotion.
The shared body is maintained but these closed-edge owner heads do not establish
useful shared-body supervision; a separate reconstruction/retention evaluation
remains necessary. Do not present this as a validated whole-body model.

The first real run caught a newly introduced float32 conversion of anchor hours
before subtraction from float64 event hours. It could create a false negative
anchor gap. Subtraction now preserves event-time dtype; a large absolute-clock
regression test covers it. This is a new bridge bug, not an explanation of old
experiments. No clamping of legitimately future events was added.

No optimizer weights saved or release changed. Full matched scoped/unscoped
training, multiple patients, held-out outcomes and intervals remain pending.
Evidence: `artifacts/native_organ_future_supervision_20260913.json`.
