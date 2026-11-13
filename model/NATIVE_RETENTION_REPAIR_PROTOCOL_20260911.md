# Native output-retention repair

Seven fixed source partitions, native glucose whole model, original32epoch
deterministic-dropout protocol. Add target-balanced SmoothL1 retention loss
with fixed weight1 against the initial model's other normalized predictions.
Use only anchor-observed channels with >=30 source-fit patients; exclude
glucose. Teacher values are cached from the same initial model and legal
inputs, never future labels. Initial outputs are imperfect references, NOT
clinical truth or validated runtime experts. This is an interference-control
experiment, not pseudo-label evidence of physiology.

Compare original glucose-only fit versus retention fit, same capacity,
initialization, partitions, selection on tune glucose MAE including epochzero.
Evaluate glucose against original, own and persistence; other-target actual
future MAE versus original and initial, patient-level bootstrap2,000, >=30
patients. Keep the initial-input mask as training eligibility, not future
measurement eligibility. Retention can constrain useful change and preserve
initial errors; report both tradeoffs, not just reduced prediction drift.

No lambda/target search on held-out results. Previous auxiliary experiment used
measured future supervision, not frozen-output retention. New event contract
and explicit retention evaluation distinguish this contrast. No runtime,
calibration, causal or promotion changes without full gate.
