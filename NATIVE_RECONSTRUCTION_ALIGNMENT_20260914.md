# Frozen Patient Alignment Diagnostic

Report: `artifacts/native_reconstruction_expanded_alignment_20260914.json`.
All original per-target MAEs reproduce within1e-7 and upstream hashes verify.
Three synthetic tests validate constant predictions, perfect predictions and
invalid-input rejection.

Without retraining, compare each patient's saved prediction with the expected
absolute error from uniformly assigning any OTHER patient's prediction for
the SAME target and SAME evaluation role. This computes all off-diagonal
pairings exactly, not a favorable random permutation. It is conditional on
this finite cohort and has no new population confidence interval.

Correct alignment has lower mean error for38/39 targets in each role.
The exceptions are BNP in evaluation and troponin I in hospital. All14
previous nominal both-role gains have positive alignment benefit in both roles.
The selected placebo is constant and has zero benefit up to floating-point
rounding (<1.2e-16).

| Target | Evaluation aligned / wrong-patient MAE | Hospital aligned / wrong-patient MAE |
|---|---:|---:|
| Albumin | 0.5494 / 1.0465 | 0.5580 / 1.0610 |
| Bicarbonate | 0.9038 / 1.1692 | 0.8591 / 1.0957 |
| Glucose | 1.3315 / 1.4555 | 1.6921 / 1.7855 |
| Creatinine | 1.6250 / 1.8789 | 1.3345 / 1.5699 |
| MAP | 0.7742 / 0.8000 | 0.8412 / 0.8515 |

Units are normalized, not physical. Predictions are not constant across
patients in this expanded run. But alignment benefit is not necessarily
enough to outperform a population center: creatinine and MAP still do not
have the prior paired population gains in both roles. Nor does this separate
measurement support/cadence from measured-value physiology, because both
travel with the donor's input. It is not an organ causal test or a clinical
personalization validation. No additional targets are promoted.

No weights, intervals or runtime changed. These diagnostics argue against
the specific explanation that all expanded outputs are merely one common
offset. They do not identify a unique remaining cause or replace repeat-split,
process-control and future-forecast evaluation.
