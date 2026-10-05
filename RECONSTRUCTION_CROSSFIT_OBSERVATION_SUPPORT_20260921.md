# Fixed Outer Observation Support

## Method

Audit the exact 247610 eligible patient-target examples from all five frozen
outer folds, all33targets. Parent dependencies, shard hashes, outer prediction
hashes, fold roles, exact labels and eligible counts are verified. Apply the
same fold-supported masking/normalization as inference. No training, selection,
prediction changes or dropped adverse patients. The original one abstention
still belongs to the parent coverage denominator.

For each independent observed variable, compute freshest measurement age at
the anchor as token age*24 plus anchor-minus-event hours. Missing sentinels are
excluded; observed numerical zero remains observed. Report available variables,
counts fresh within1h/6h, median age, event counts, history span, normalized
label distribution, population error and fixed real-minus-process error.
Token-entry counts are NOT deduplicated clinical measurement counts.

Only aggregate group/component statistics are saved. Component ordinals are
local to the target and carry no patient identifiers. These summaries are
diagnostic, not new routing features or a matched-state comparison.

## Findings

Group means (selected / broad):

| Target | Variables seen | Variables fresh <=1h | Mean within-person median age (h) | History span (h) |
|---|---:|---:|---:|---:|
| Glucose |16.31 /15.14|2.95 /6.28|10.93 /8.30|30.87 /26.31|
| BUN |15.16 /14.78|9.72 /10.59|3.46 /2.35|25.49 /23.15|
| Creatinine |17.24 /16.86|12.01 /12.63|1.95 /1.78|29.05 /24.93|
| Hematocrit |13.88 /12.44|4.84 /3.83|9.14 /11.41|28.60 /25.49|

Glucose selected examples have more distinct variables and longer histories,
but substantially fewer current observations. The latest target label mean
also differs (+0.75849 versus+0.12948 in parent normalized units). It is not
valid to equate longer history with more informative current physiology.
This association does not prove that stale history causes the error.

The larger adverse creatinine component (selected156 / broad2463patients)
provides a counterexample to a universal freshness explanation:

| Metric | Selected | Broad |
|---|---:|---:|
| Variables seen |14.250|14.646|
| Fresh variables <=1h |10.199|10.562|
| Mean median measurement age (h) |1.765|1.987|
| History span (h) |27.150|27.567|
| Mean normalized truth |0.561|0.558|
| Real-minus-process MAE |+0.10288|-0.01098|

Here selected inputs are not older on this summary, and label means are close,
yet model benefit changes sign. Equal means/counts do not establish identical
feature identities, distributions, correlations, timing or physiology. This
does NOT prove irreducible noise or a unique encoder defect.

## What This Changes

Do not globally discard old events or change a single freshness threshold and
declare the transfer problem solved. Glucose warrants observation-timing
analysis; creatinine needs within-component feature/label relation analysis
beyond support counts. No new model should be selected on these outer outcomes.
The next controlled repair must use fitting data for any freshness weighting or
input selection and retain a comparison with the unchanged architecture.

Three tests verify anchor-relative age, missing versus observed-zero semantics,
and rejection of future events. Source report:
`artifacts/native_reconstruction_crossfit_observation_support_20260921.json`.
No runtime, model weights, promotion, interval or causal claim changed.
