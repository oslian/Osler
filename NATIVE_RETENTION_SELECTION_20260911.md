# Source-only retention reference selection

Prespecified rule for this next contrast: initial reference must beat
persistence with patient-paired95% upper confidence bound below0, using only
source tune patients, minimum30 patients per target. Exclude primary glucose.
No hospital/evaluation rows, final retention predictions, or future test-name
strata are used to choose references. Source tune remains development, not a
final test. Criterion is exploratory per target, not familywise confirmation.

Executed selection: `artifacts/native_retention_selection_20260911.json`.
Selected bicarbonate and potassium7/7 partitions, magnesium6/7 (not seed7).
No BUN/creatinine/calcium/phosphate/hematocrit references pass this rule.
This does not establish those targets lack signal; it assesses the INITIAL
candidate reference against persistence, not a disease or other expert source.

The result supports replacing indiscriminate retention with reference-quality
selection, but accuracy of the replacement is NOT yet measured. Next train
the same native model/32epochs with the same weight1 retention loss and only
these source-selected targets, retaining source-fit support and anchor masks.
Compare with glucose-only and uniform-retention controls. Score actual future
values across all supported targets, including BUN/creatinine, not only the
selected ones. Report primary and retention tradeoffs separately.

Inputs/selection are frozen before this next accuracy run. No new model fit,
runtime, interval or promotion change. Reused cohorts and uncertain historical
pretraining membership remain limitations.
