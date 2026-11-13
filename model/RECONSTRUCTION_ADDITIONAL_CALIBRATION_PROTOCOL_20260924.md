# Frozen Multi-Component Offset Repair

This follow-up is specified after the outer diagnosis was viewed. Existing
outer patients are reused validation, not a new untouched confirmation cohort.
Do not claim independence from research design merely because fitting excludes
them. No runtime promotion or uncertainty calibration is part of this test.

Freeze all epoch8 encoders, fit-only scalers, roles and47targets. Produce
predictions only for each fold's preassigned CALIBRATION role with that fold's
own model. No substitute predictions from other folds unless model state,
scalers, baselines and exact patient role equivalence are verified. Fold2may
reuse fold0outer predictions under the already certified identical fit.

For EACH of real, process and population, estimate one additive raw-unit
offset per target/fold using calibration truth minus prediction. Minimize
component-equal absolute residual error: each component has total weight1,
divided equally among that target's patients. Use the weighted median (lower
median at exact ties). No slope, hyperparameter sweep, outlier removal, clipping
or target-specific choice. Every target receives the same procedure.

Require at least30calibration patients in each of at least3components for that
target. Otherwise offset0with explicit insufficient_support, not a fitted
correction. When supported, include all calibration components in the weighted
median, not just large ones. Never fit outer/hospital-specific offsets.

All calibration inference must complete before assessing corrected outer
predictions. Compare corrected real against uncorrected real AND corrected
process/population on identical outer rows. Report all47raw-unit patient and
component MAEs, paired CIs, normalized macros, support/fallback and every
component. Improvement screen requires negative means and p<.05/47 against
all3comparators at both levels; this incremental screen is not promotion.
Also retain the original uncorrected report; do not replace a failure silently.
No requirement that each target must pass; no further tuning on this result.
