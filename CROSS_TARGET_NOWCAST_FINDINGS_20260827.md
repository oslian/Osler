# Cross-Target Current-State Completion Findings

Date: 2026-08-27

## Question

Can related physiological measurements estimate a currently unknown target,
and thereby increase observed-state coverage without fabricating future values?

## Design

- Point models are trained only on eICU.
- Inputs are other current physiological values only. Target values, target
  ages, workflow fields, future values, and deterministic/near-deterministic
  sibling groups are excluded.
- MIMIC-IV supplies seven patient-disjoint validation splits and 90% conformal
  calibration/evaluation splits.
- A target must significantly beat the eICU population median and a
  capacity-matched shuffled-target placebo in all seven splits.
- The care-unit gate requires improvement for at least 90% of supported
  patients and forbids regression in any care unit holding at least 10% of
  supported patients.
- Current-state completion, future forecasting, and causal treatment effects
  remain separate tasks.

## Result

Eighteen shared eICU/MIMIC targets were evaluated. Two passed the full point,
placebo, care-unit, and interval gates.

| Target | Cross-target MAE | Population median MAE | Placebo MAE | Mean patient delta | Mean 90% coverage | Minimum care-unit patient coverage |
|---|---:|---:|---:|---:|---:|---:|
| bicarbonate | 3.432 | 3.645 | 3.679 | -0.181 | 0.901 | 0.916 |
| heart rate | 13.980 | 15.025 | 15.338 | -1.197 | 0.902 | 0.998 |

The runtime bundle is `models/cross_target_nowcast.joblib`. A value is emitted
only when at least four related measurements are available. Observed values
always take precedence. Unvalidated targets remain missing.

## Negative Results

Several targets improved in all seven patient splits but failed the care-unit
domain gate: MAP, pH, platelets, potassium, respiratory rate, and temperature.
They are not enabled. Bilirubin, BUN, creatinine, glucose, lactate, oxygen
saturation, sodium, urine output, and WBC did not pass the seven-split point
gate.

## Runtime Boundary

The completion is exposed as `current_estimate` with explicit provenance and a
current-state interval. It does not overwrite `observed_value`, does not turn a
missing future forecast into a point prediction, and is not fed back into the
future JEPA until a separate cascade-error audit proves downstream benefit.

This is factual research output only. `causal_claim_allowed` remains false.
