# Independent Calibration Offset: Fixed Protocol

This repair is motivated by a posthoc diagnostic, not a new untouched test.
In the adverse creatinine selected component, real rank correlation .378
exceeds potassium-removed .301, despite worse MAE. Their retrospective
median-offset-oracle MAEs are1.31788 and1.31806, respectively. This oracle uses
held-out outcomes and is NEVER a fitted deployable calibrator.

The actual experiment uses all33targets/all5folds. Each frozen real encoder
predicts its preassigned `encoder_tune` patients (neither encoder fitting nor
outer test). For each target, estimate truth-minus-prediction median with equal
total selected/broad group weights, using existing equal_group_median. Require
at least30calibration patients in EACH group; otherwise keep zero offset with
explicit unsupported status. No tuning of coefficients, thresholds or variants.
One offset per target/fold applies to ALL outer patients, independent of their
group, hospital, outcome or input values. This isolates transferable intercept
calibration and cannot repair ranking.

Compare corrected versus unchanged, process and population on exactly the
same eligible outer patients. Preserve abstentions. Report all33targets and
all/selected/broad scopes with patient/component paired means and the existing
descriptive screen (including unchanged model as a comparator). Unsupported
calibration is not a new pass. Never pick the better model after outer scoring.

This differs from earlier BUN affine tests: freshly fitted patient/component-
independent encoders, fold-local normalization, all-target source-calibration
and outer evaluations. It is NOT a new external lockbox. No runtime promotion
follows automatically, even if the descriptive screen improves. Weights, data
roles and deployed intervals remain unchanged.
