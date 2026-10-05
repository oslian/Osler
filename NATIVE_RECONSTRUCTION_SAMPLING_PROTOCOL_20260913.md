# Matched Multi-Target Sampling Pilot

Fixed seed7 partition, deterministic RNG20260913:32fit/16tune/16evaluation/
16hospital patients, disjoint roles. Raw-event native anchors, fresh<=1h truth,
full target/sibling masking. One fixed reservoir example per patient/target;
retain source event counts for weighting. No future labels select samples.
Targets need5fit and3tune patients. These support minima are pilot eligibility,
not statistical sufficiency or promotion. All eligible targets share one body.

Two identical initial models: event-frequency weighted reservoir draws versus
uniform target then uniform patient draws. The former approximates event
frequency but does not replay every event. Readout starts at fit target median,
weights zero. Same8epochs x64steps, AdamW1e-4, clip1, SmoothL1, eval-mode
dropout disabled for matched reproducibility. Whole model trainable, no new
future-JEPA objective in this current-state reconstruction task.
Select0..8 by target-equal tune normalized MAE. Report every target's normalized
MAE on heldout roles and paired patient bootstrap vs population and other arm.
Single source split, small pilot, reused heldouts; no promotion or intervals.
Store candidate weights/private samples locally only; runtime unchanged.
