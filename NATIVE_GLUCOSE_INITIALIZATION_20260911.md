# Initialization versus fine-tuning

Frozen diagnostic, no new fits. Reconstructed the initial model on the same
native inputs, allowed channels, roles and identity decoder used by the 21-fit
glucose trial. Paired patient bootstrap, 2,000 draws. Aggregate:
`artifacts/native_glucose_initialization_20260911.json`.

| Patient-equal MAE mg/dL | Own initial | Whole initial | Own final | Whole final |
|---|---:|---:|---:|---:|
| Fit | 36.3297 | 37.2041 | 34.0378 | 32.2889 |
| Tune | 35.5665 | 35.8737 | 33.5358 | 33.7316 |
| New source | 35.7467 | 36.2445 | 33.9762 | 34.8408 |
| Hospital | 48.3919 | 46.0267 | 41.4025 | 42.4007 |

Median across overlapping seeds; not a paired difference of medians.
Initial whole-vs-own significant gain/harm: fit0/5, tune0/3, newsource0/2,
hospital7/0. Final whole-vs-initial gains: fit7, tune5, newsource2, hospital6,
with no significant harm. Own final-vs-initial gains7/5/2/7, no significant harm.
Final whole-own still fails significant gain on tune/newsource/hospital.

## Interpretation correction

The native fit does not simply damage every whole forecast: it improves
hospital absolute error in6/7. The hospital relative ordering reverses because
own fine-tuning improves more. Initial whole context already harms some source
comparisons but helps hospital comparisons. Thus there is domain-dependent
initialization behavior as well as learned fit-to-new-patient reversal.
The final fit gap remains real, but it alone cannot establish that fine-tuning
destroyed an otherwise universally useful cross-organ representation.

This is not a from-scratch comparison or proof of a superior initialization.
The same inherited weights are evaluated under different input restrictions;
they are not separately pretrained own/whole encoders. Metadata cannot certify
historical pretraining membership. New/source and hospital cohorts are reused,
potassium-selected and do not establish universal glucose/whole-body behavior.
No runtime, interval, causal or promotion changes.
