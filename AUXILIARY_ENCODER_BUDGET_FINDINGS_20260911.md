# Longer Auxiliary Supervision: Learning Is Not Transfer

The64epoch seven-seed source-only trial completed all21 fits. All first8
epoch logs exactly reproduce the three-seed pilot; all64 single-target logs
exactly reproduce the preceding2250/64 whole-unbounded experiment. Same queries,
patients, architecture, initialization, losses and optimizer. No external data
was evaluated and no runtime model or calibration changed.

## Primary Glucose Result

Selected source-tune patient-equal MAE, mg/dL:

| Seed | Single target | Real auxiliary | Shuffled auxiliary |
|---|---:|---:|---:|
|7|37.0549|37.0226|37.1173|
|19|35.1903|35.1124|35.2591|
|31|36.1447|36.2401|36.0244|
|43|35.1978|35.3102|35.2659|
|59|34.1241|34.1300|34.2180|
|71|38.1734|38.3273|38.2503|
|83|37.3943|37.3931|37.4771|

Real auxiliary significantly improves single-target0/7 and shuffled0/7.
No significant single-target harm, but seed31 is significantly worse than
shuffled (error delta0.2157,95% CI[0.0623,0.3598]). Real auxiliary selection
epochs64/64/59/61/59/45/64; a64epoch budget is not proof of convergence.
The effect remains small and inconsistent despite longer optimization.

## Did Other Targets Learn?

Post-training diagnostic uses the SAME source-tune patients, each target's
measured-pair support, and checkpoints selected for glucose. These are not
new external tests or independently selected best auxiliary models.

Significant paired improvements, with supported-seed denominator (>=30patients):

| Target | vs single-target training | vs shuffled auxiliary | vs persistence |
|---|---:|---:|---:|
|Bicarbonate|7/7|3/7|0/7|
|BUN|7/7|0/7|0/7|
|Calcium|7/7|0/7|0/7|
|Chloride|6/7|3/7|2/7|
|Creatinine|3/7|0/7|0/7|
|Hematocrit|7/7|0/7|0/7|
|Hemoglobin|6/7|1/7|1/7|
|Potassium|7/7|4/7|6/7|
|Sodium|7/7|3/7|1/7|
|Anion gap|3/4|0/4|0/4|
|Platelets|4/6|0/6|0/6|
|WBC|1/6|1/6|0/6|

Magnesium and phosphate have no supported tune split, not seven failed tests.
For example potassium median MAE: single0.4791, real0.3699, shuffled0.4275,
persistence0.4391. BUN real3.4338 versus shuffled3.4143 and persistence3.2965:
better than an unmaintained auxiliary head is not proof of useful dynamics.

These findings show some auxiliary learning, but not universal benefit,
cross-organ attribution or a promotion. Multiplicity, repeated selection
patients, small target-specific supports and pretraining overlap limit inference.

## Gradient Diagnostic

At the seven selected real-auxiliary checkpoints, sample three patient-uniform
source-fit batches each. Compute primary and auxiliary gradients on the shared
context parameters, not on unrelated head parameters.5/21 gradient cosines
are negative; median weighted-auxiliary/primary norm ratio0.2544. The signal is
not disconnected or uniformly negligible, and pervasive opposing gradients
are not established. This is a local diagnostic at selected checkpoints, not
the complete optimization trajectory or a causal proof about training harm.

## What This Localizes

Extra measured tasks can be learned without yielding consistent improvement
in the primary glucose task. Thus missing auxiliary supervision alone is not
an experimentally supported complete repair. The first observed failure remains
cross-patient generalization, not exclusively external harmonization.

Do not infer that every shared-body mechanism has failed: this trial retained
the base world-model and supervised one future window. It did not train a
persistent organ-state trajectory or prove information absence. No further
blind increase of auxiliary weight is justified by these data. Any next change
must distinguish conditional predictive value from simple auxiliary fitting.

Evidence: `artifacts/encoder_auxiliary_budget_20260910.json`,
`artifacts/encoder_auxiliary_learning_20260911.json`.
Code: `scripts/train_auxiliary_encoder.py --extended`,
`scripts/audit_auxiliary_learning.py`. Protocol was fixed on2026-09-10; completed
and inspected on2026-09-11. Weights and patient predictions remain private.
