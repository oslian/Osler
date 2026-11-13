# Event Representation Versus Persistent State: Completed Diagnostic

Fixed protocol, all47targets/fivefolds,235 paired target-fold runs,2350 fitted
readout models. Frozen epoch10 encoder; no encoder, state or runtime updates.
At most512fit/256selection patients per cell, chosen by outcome-independent
hash. Completed59629 query occurrences,5703 unique patients. Target support
after averaging repeated folds: minimum337patients and3source components.

Each latent arm has64 coordinates plus the SAME159 process covariates; raw
has265 dimensions and process159. Each gets separately fitted HGB and Ridge.
These are reused development roles, not a fresh external test or full-cohort
promotion. Current target and all siblings/own history are masked.

## Results

Normalized target-macro MAE (not accuracy percentages):

| Readout/input | Patient equal | Component equal |
|---|---:|---:|
| HGB event mean | .911521 | .913828 |
| HGB last event | .895912 | .894731 |
| HGB persistent body | .885501 | .880512 |
| HGB raw | .869179 | .871693 |
| HGB process | .949792 | .947550 |
| Ridge event mean | 1.246774 | 1.215507 |
| Ridge last event | 1.223184 | 1.199495 |
| Ridge persistent body | 1.216441 | 1.203538 |
| Ridge raw | 1.260501 | 1.242880 |
| Ridge process | 1.219957 | 1.197003 |

All eight prespecified47-family corrected comparison screens have zero passes.
This does not mean the means are equal or all representations contain no signal.
HGB body has lower patient mean error than event mean in41/47targets, than last
event in40/47. Component counts34/47 and33/47. Ridge body is lower than event
mean in36/47patient and34/47component means. These counts are descriptive mean
signs, NOT corrected significant wins for body (reverse tests were not added).

Examples, HGB patient-equal normalized MAE:

| Target | Event mean | Last event | Body | Raw |
|---|---:|---:|---:|---:|
| BUN | .664454 | .642882 | .630306 | .614154 |
| Creatinine | .875876 | .819914 | .824523 | .806442 |
| Glucose | .788397 | .799069 | .792428 | .793373 |
| Bicarbonate | .624782 | .621485 | .614139 | .599738 |

Glucose's small mean benefit and creatinine's last-event mean benefit do not
pass the prespecified screens; neither is selected for serving.

## What Changed In The Diagnosis

The simple repair hypothesis, replacing persistent aggregation with event mean
or the last event, is not supported. The current persistent state often makes
these fixed event representations MORE usable by the controlled readouts.
Thus E16's old bounded7 BUN body-dynamics harm cannot be generalized into
universal state destruction or copied onto this epoch10 model.

Raw HGB remains better overall, whereas Ridge raw does not beat Ridge body
overall. This is further readout dependence, not irreversible encoder-loss
proof. Event averaging discards order and detailed timing; last event discards
coverage. Consequently failure of those summaries cannot locate the first
information loss at the event encoder. No arbitrary extra head sweep follows.

The unresolved comparison is information preservation under a common,
variable-aligned observation readout objective before/after aggregation, with
proper fit-only fitting and nontrivial controls. This is a question, not a
completed test or authorization to bypass the existing protocol gate.

## Verification And Limits

- Every sampled example's raw summary and truth matched original cache; first
  sample per target/role reproduced the original model to1e-7 (470 checks).
- Five cache/weight/role contracts verified; per-row digests checked, complete
  target family, deterministic query IDs, provenance, truth and scores rechecked
  in final assessment. Private data remain ignored, no identifiers published.
- All query predictions finite. StandardScaler emitted sqrt warnings for some
  fits; scaler objects were not retained, so no post-hoc claim about their exact
  variance values is made. Warnings did not produce nonfinite predictions.
- Twelve tests pass: six new stage/scoring tests plus six existing fixture
  tests. They do not establish clinical performance or resolve population shift.
- No saved deployable probe checkpoints, intervals, runtime promotion, outer
  evaluation or treatment-causal claims. Bounded sample is diagnostic only;
  no claim that full-data training must have the same result.

Evidence: `artifacts/native_reconstruction_stage_fold{0..4}_20261003.json`
and `artifacts/native_reconstruction_stage_assessment_20261003.json`.
All five training processes and final assessment exited0.
