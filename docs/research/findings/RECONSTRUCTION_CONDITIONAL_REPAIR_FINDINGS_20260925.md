# Conditional Correction: Rejected, Failure Localized Before Outer Transfer

Completed the fixed protocol on all1,800,214outer patient-target pairs,
73,670people,47targets,fivefolds. No outer support/labels set input inclusion,
scaling, penalty, weights or model selection. No activation was selected.
Calibration patients fit the corrections; source leave-component-out refits
all preprocessing. Reused outer validation is not a fresh lockbox.

## Outer Results

| Normalized target-equal MAE | Unchanged | Median offset | Process-context correction | Numeric-context correction |
|---|---:|---:|---:|---:|
| Patient |0.925567|0.917067|1.126057|1.119574|
| Component |0.901157|0.897123|1.129490|1.117577|

ZERO targets satisfy the prespecified incremental descriptive screen versus
allthree references at both patient and component levels. Numeric correction
has lower patient mean MAE on7targets and component mean on6; no universal
replacement is justified. The process-context arm still uses the original
numeric model's frozen prediction; it is not a process-only encoder.

Examples of raw-unit patient MAE, unchanged -> numeric correction:
MAP13.26625 ->13.18541 (median offset12.82157 is better);
glucose37.78697 ->44.65429; creatinine0.63340 ->0.72086;
albumin0.38851 ->0.40403; heart_rate14.17640 ->14.06611, but
heart-rate equal-component MAE worsens14.16567 ->14.41952.

## First Demonstrated Failure

163target-folds have complete supported source leave-component-out assessment.
Numeric correction worsens MAE in138of163. Thus this repair's first observed
generalization failure is WITHIN source calibration hospitals, not exclusively
an external database problem. These source warnings were reported, not hidden
by selecting a favorable target after seeing outer results.

An identical-fit replay of all235targetfolds reproduces saved outer predictions
within1e-10absolute/1e-12relative tolerance. Of189supported fits,35reduce fitting
MSE yet increase fitting MAE. This directly demonstrates an objective mismatch
for this squared-loss correction. It does not establish a bug in the prior
SmoothL1 encoder or explain all earlier failures.

Of the138source-CV harms:
- 27already have worse in-sample MAE.
- 111improve in-sample MAE, then worsen when a source component is withheld.

Therefore switching to an MAE-aligned objective alone cannot be assumed to fix
the dominant transfer gap. Finite-sample overfitting, component-specific
relationships and insufficient state conditioning remain distinguishable
hypotheses, not a proven unique clinical cause.

## Interpretation And Next Repair Boundary

Observation identity shifts are real, but a flexible conditional correction
trained on these snapshots does not automatically transport. The larger
cohort does not overturn the previous small-cohort raw-ridge negative result.
Stop treating extra residual correlations as proof that another readout will
improve held-out errors.

A next repair must first survive source leave-component-out comparisons;
outer retesting should not be the first check. A robust/median objective tests
the35fit-level mismatches, while stable coefficients or stronger component
regularization address the111fit-to-held-component reversals. Neither has been
validated by this experiment. Do not lower support or promotion thresholds.

Five new unit tests pass, including a skewed-label counterexample where MSE
optimization worsens MAE before transfer. Runtime emitted sklearn weighted
variance sqrt warnings on some nearly constant features; all resulting model
inputs/predictions used by Ridge were finite, and fixed-fit outputs reproduce.
Warnings are recorded, not evidence of the historical root cause.

No encoder weights, runtime, interval, registry or forecast changed. Current
masked reconstruction only; no formal promotion. Unequal feature dimensions
are not a capacity-matched placebo, and crossfit models overlap in fitting.

Artifacts:
- `artifacts/native_reconstruction_conditional_repair_20260925.json`
- `artifacts/native_reconstruction_conditional_fit_objective_20260925.json`
