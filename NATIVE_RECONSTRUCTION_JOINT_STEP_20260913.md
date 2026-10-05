# Multi-Target Examples To One Shared Body Update

Integration test exposed a token-width mismatch: builder returned the basic
measurement schema while the native state bridge expects six extra event
channels. Reused append_events with six zero channels for this explicitly
measurement-only task. No invented treatments or altered clinical values.

Fifteen tests pass. The new end-to-end test runs a masked original-event example
through NativeBodyReconstruction and verifies finite, nonzero body gradients.
Earlier builder report binds pre-repair source SHA256
d8cb98c77db3a0f525736c97b94d6c80ad6c82e0d9bad330749841f4e0cd5a9d;
it remains evidence of sample generation only, not compatible neural ingestion.
Do not silently rewrite its bindings to the new source.

A real-source one-step smoke uses the first four distinct target examples from
the first64 anchors of the prior source-only pool. CPK, INR, PTT and heart_rate
each have finite body-update gradients(.2012,.1553,.1918,.0784). Their mean loss
backpropagates through the same body-state model; one optimizer step changes
body update parameters (L2 .009089). Different masked examples necessarily
produce different inferred states; this is shared parameters and trajectory
machinery, not evidence of one unmasked state reconstructing every target.

The report's patients8/anchors64 describe the input candidate pool, not eight
independently trained patients or64 consumed examples. Only four examples were
used for the step. No heldout evaluation, efficacy, JEPA future-loss comparison,
saved weights or promotion. Runtime/registry/calibrators unchanged.

Evidence: artifacts/native_reconstruction_joint_step_20260913.json.
