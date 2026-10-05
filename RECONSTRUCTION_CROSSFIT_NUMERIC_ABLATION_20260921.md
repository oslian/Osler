# Frozen Creatinine Numeric Ablation

## Completed Scope

All five frozen fresh-encoder folds, all10192eligible creatinine patients:
530selected/9662broad. Original predictions reproduce exactly (maxdifference0).
Replace each of three input variable histories with fitting-fold location,
separately. Masks, ages, times, other numbers and model parameters unchanged.
All patients retained. Private per-fold predictions are checksum-bound; outputs
are aggregate only. This follows the fixed diagnostic protocol, not a feature
search or a confirmatory external validation.

## Overall Result

Paired MAE change versus unchanged real model; positive means WORSE. Units are
the original common normalized target scale, not raw clinical units.

| Numeric history replaced | Selected patient mean | Broad patient mean | Selected component mean | Broad component mean |
|---|---:|---:|---:|---:|
| Bicarbonate |+0.001414|+0.010249|+0.007588|+0.009826|
| Potassium |+0.003186|+0.021052|+0.009480|+0.029217|
| Hemoglobin |+0.002251|+0.005191|+0.001766|+0.002072|

All three removals numerically worsen both groups overall. None constitutes a
general repair. Selected predictions still lose process/population controls.
No statistical significance or promotion claim is attached to these deltas.

## The Previously Adverse Component

Selected156patients / broad2463patients:

| Numeric history replaced | Selected MAE change | Broad MAE change | Selected remaining excess error versus process |
|---|---:|---:|---:|
| Bicarbonate |-0.018624|+0.012168|+0.084251|
| Potassium |-0.078120|+0.001648|+0.024755|
| Hemoglobin |+0.004195|+0.009371|+0.107070|

Removing potassium helps this subset more than removing bicarbonate, despite
the previous marginal potassium-creatinine correlations being nearly equal
across groups (.260/.264). Thus marginal correlation stability does not establish
that the learned numerical response is appropriate, or that a variable's
conditional contribution transfers. Even this posthoc best subset intervention
does not beat process or population. Do not install a component/group rule.

Median substitution may be off-distribution and this is not a physiological
intervention. The results establish dependence of the frozen function, not a
causal mechanism or proof that potassium is biologically unhelpful. They do not
separate response magnitude, interactions, calibration, and fitting error.

## Decision

Reject a global deletion of any of these three inputs. Reject the simpler
claim that unstable marginal bicarbonate correlation is the sole defect.
The model uses useful numbers overall but can use them adversely in a subset;
the subset's errors are not fully removed by suppressing one channel. Any
conditional repair must be learned using source-fit data and independently
evaluated, not selected from this subgroup's known errors.

Two unit tests pass (only the requested numeric column changes; paired-error
sign), and all10192original predictions exactly reproduce. No runtime, weights,
registry, interval or gate changed. Report:
`artifacts/native_reconstruction_crossfit_numeric_ablation_20260921.json`.
