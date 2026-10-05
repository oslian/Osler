# Variable Identity And Relation Diagnosis

## Fixed Scope

All33targets,247610eligible frozen outer examples. Same parent predictions,
patient shards, fold-specific supported masks and scaler bindings. Check exact
labels and outer roles, preserve missing values and observed zero, and assert
the target plus all contract siblings remain absent. No models or features are
selected and no predictions are changed.

For each input variable, use its last available observation, convert its value
back to the common parent normalized scale, and add elapsed time to its stored
measurement age. Report observation fractions, age, and Spearman correlation
with the masked target for selected/broad patients and each hospital component.
Correlation is omitted below30observations or for constant values. This support
threshold is only a reporting rule, not a promotion rule. Correlations condition
on being observed and are unadjusted, exploratory associations, not causal or
statistical-significance claims. No claim that a univariate correlation equals
useful conditional predictive information.

## Why Similar Counts Are Not Similar Inputs

In the previously identified adverse creatinine component, aggregate support,
age and label means looked close. Individual relationships differ:

| Input for creatinine | Broad observed / rho | Selected observed / rho |
|---|---:|---:|
| Potassium |2416 /+0.2602|155 /+0.2641|
| Bicarbonate |2426 /-0.2641|155 /-0.0394|
| Hemoglobin |2035 /-0.1395|121 /-0.3104|
| Phosphate |294 /+0.5576|15 /not reported|

Potassium and bicarbonate both have median measurement age0.7667h in both
groups. Therefore freshness alone does not explain why their association
patterns differ. Sampling variability, different case mix, conditional
relationships and measurement selection remain possible explanations.

Phosphate has a strong descriptive association in the broad group but is
available in only15selected patients of this component. A count of all available
variables hides this difference in WHICH informative variable was observed.

Across all patients, creatinine-phosphate rho is+0.4395broad/+0.4514selected,
but support is2522/9662 versus185/530. This is not evidence that phosphate alone
can reconstruct creatinine or that the encoder failed to use it.

## Glucose Timing Differences Are Variable-Specific

Median age of observed latest input (broad / selected):

| Input | Age (hours) | Glucose rho broad / selected |
|---|---:|---:|
| Creatinine |0.950 /7.792|+0.0895 /+0.0900|
| Potassium |0.950 /5.175|+0.0584 /+0.0051|
| Bicarbonate |0.950 /5.317|-0.0403 /-0.1428|
| Hemoglobin |3.600 /7.425|-0.0229 /+0.2073|

These associations are not uniformly strong, nor uniformly stable. Merely
presenting additional past values cannot guarantee an identifiable current
glucose value when its own history and siblings are deliberately hidden.
This does NOT negate the separate forecasting task where its own history is
available. Do not mix masked reconstruction and forecast capability claims.

## Repair Implications And Limits

The remaining issue is finer than missing count or median age: the identity
and conditional usefulness of available observations varies by group. This
does not yet prove the model relies on an inappropriate relation. A controlled
source-trained intervention must establish that before changing the encoder.

The next discriminating check is a matched, frozen-model input ablation of the
identified relation on development predictions, preserving masks/times in a
separate numerical-control arm. This can test model reliance, but cannot itself
prove a new training method transfers. Any feature selection, conditional
weighting or retraining must use fitting data, followed by separate validation;
these outer examples are now diagnostic development data.

Three tests pass for last-available selection/inverse scaling, missingness and
sample support, and rank behavior. All outer target/sibling leakage assertions
pass. Report:
`artifacts/native_reconstruction_crossfit_variable_relations_20260921.json`.
No runtime, weights, gate, interval or promotion changes.
