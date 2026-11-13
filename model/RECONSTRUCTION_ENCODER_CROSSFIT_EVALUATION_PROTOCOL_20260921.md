# Fixed Outer Evaluation

Do not execute inference until all five folds/two arms finish their fixed eight
epochs (the explicitly verified equivalent control counts as the same fitted
function, not independent training). Verify checkpoint, input and dependency
hashes. Use only each patient's assigned outer-test model/scaler and report all
33 trained targets. Preserve unsupported/empty-history rows in coverage.

Compare true-value reconstruction by real encoder, process-only encoder and
the fold's equal-group fitting-label median. Target and algebraic siblings remain
masked, including their history. No persistence comparison is appropriate for
this task because the target's own value/history is intentionally hidden.
Invert fold scaling before errors so all predictions share original normalized
units. These are not raw clinical units and not future forecast scores.

Publish all/selected/broad group metrics, patient-equal MAE, site-equal MAE and
complete-history component-equal MAE. Each target has at most one sample per
patient. Against both controls, report paired mean error differences and fixed
5000-resample95% bootstrap intervals for patient means and equal-component means.
Use seed20260921 plus fixed scope/target offsets; do not tune seeds to outcomes.
Intervals are descriptive conditional-on-fitted-predictions and are unadjusted.
Overlapping crossfit training sets are not accounted for by this bootstrap;
they cannot certify the entire learned algorithm's uncertainty.

Reuse the earlier conservative screen:>=30patients,>=3components, patient and
component mean improvement, one-sided t-testp<.05/39 against BOTH controls.
Call this a descriptive development screen, not promotion. Also show whether
both selected and broad scopes screen positive; never omit their contradictions.
No hyperparameters, targets, weights or gates are changed after scoring.
Patient and component bootstrap alone do not establish cross-database validity,
conformal coverage, clinical usefulness or causal understanding.
