# Source-Only Offset Activation

New follow-up, not retrospective replacement of the unconditional experiment.
All encoders, splits, predictions and47targets remain frozen. For each arm and
target/fold, fit the already specified component-equal offset on calibration.
Before using it, leave each calibration component out and estimate the offset
without it; measure its MAE change on the omitted component.

Enable only if ALL omitted-component fits have the original support, at least
4components are assessed, mean MAE difference is negative and the one-sided
component t-test has p<.05. No threshold sweep or target-specific exceptions.
This p-value is a fixed selection heuristic, not a valid independent promotion
test: these source fits overlap. No multiplicity claim for activation itself.
Apply the identical selection rule to physiology, process and population arms.
Disabled correction is exactly zero, preserving the uncorrected prediction.

Evaluate all outer rows against unchanged physiology and identically gated
controls using the existing incremental p<.05/47 patient/component screen.
Also report comparison with unconditional offsets, active counts and helpful/
harmful enabled corrections. This is reused validation after diagnosis, NOT
untouched confirmation; no future forecast, interval or runtime promotion.
