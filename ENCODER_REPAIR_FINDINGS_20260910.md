# Controlled Decoder Repair Results

Completed24 fits: three seeds x own/full x bounded/unbounded x standard/balanced
JEPA pressure. Protocol: `ENCODER_REPAIR_PROTOCOL_20260910.md`.
Aggregate: `artifacts/encoder_repair_20260910.json` (complete=true).
Training settings, data support and individual checkpoint hashes are in JSON.

## Point Results

Median patient-equal MAE over seeds, glucose mg/dL; lower is better.

| Input / decoder | eICU hospital | MIMIC |
|---|---:|---:|
| Persistence | 45.878 | 37.449 |
| Own bounded | 42.112 | 34.536 |
| Own unbounded | 38.273 | 32.070 |
| Own balanced | 42.210 | 34.644 |
| Own unbounded + balanced | 38.279 | 32.408 |
| Whole bounded | 42.386 | 36.091 |
| Whole unbounded | 38.339 | 35.461 |
| Whole balanced | 42.326 | 35.412 |
| Whole unbounded + balanced | 38.485 | 34.105 |

Counts of paired patient-bootstrap95% CIs strictly below zero, out of3:

| Contrast | eICU hospital | MIMIC |
|---|---:|---:|
| Own unbounded vs own bounded | 3 | 3 |
| Whole unbounded vs whole bounded | 3 | 1 |
| Whole balanced vs whole bounded | 1 | 3 |
| Whole combined vs whole bounded | 3 | 2 |
| Own unbounded vs persistence | 3 | 3 |
| Whole combined vs persistence | 3 | 3 |
| Whole vs own, any matching mode | 0 | 0 |

The seeds reuse the same held-out patients; these are not three independent
external cohorts. Intervals are exploratory unadjusted comparisons, not a
completed multiple-comparison-aware promotion gate. MIMIC has been inspected
repeatedly and is validation, not a pristine lockbox.

## What Was Fixed / What Remains

Replacing only tanh with identity (same parameter count, same0.75 multiplier)
and retraining removes a known output-range restriction. Own-history error
falls about9.1% hospital and7.1% MIMIC relative to the bounded medians. The
matched bounded controls reproduce the earlier end-to-end pilot exactly.
Thus the hard limit was not merely a theoretical concern: its removal under
this training protocol yields reproducible point improvement. This does not
say every large change was predictable, or every historical model was capped.

The same change helps whole-body source-hospital error but is not consistently
beneficial on MIMIC. Additional context remains the separate transfer problem.
Increasing JEPA weight helps whole MIMIC modestly while not improving own; loss
balance alone is not a sufficient remedy. Most source-derived weights hit the
prespecified20 maximum (one is18.393); this is a fixed bounded weighting rule,
not proof that gradients stay balanced throughout training.

No universal cross-organ gain, complete root-cause isolation or promotion is
claimed. Still missing:7seed confirmation, care-unit/time gates, new compatible
interval calibration, negative-transfer/retention evaluation and pristine data.
Identity affects the candidate decoder globally, but only glucose was trained
and evaluated. Other targets are NOT qualified for this decoding change.

## Implementation / Reuse

`JointWholeBodyJEPA.value_delta_activation` is the explicit activation module.
Serving defaults to `nn.Tanh()` and has unchanged numerical behavior. Candidate
training replaces it with `nn.Identity()`. Both have no parameters; therefore
state_dict alone CANNOT identify the decoder. Saved candidate checkpoint
`value_delta_activation` metadata is mandatory when reconstructing it:
start from the identical base architecture, set the indicated activation,
then strictly load `state_dict`. Never attach the release's old calibrator.

Weights remain local under `/private/tmp/encoder_repair_weights_20260910`.
No row-level predictions were committed. No release weights, registry,
calibrators or serving policy changed.293 repository tests pass.

Next: confirm the amplitude repair with the full gate while continuing to
isolate source-support/encoder/forecast-path transfer. Do not repeat a
weight-only JEPA sweep as if the single-loss-pressure hypothesis were untested.
