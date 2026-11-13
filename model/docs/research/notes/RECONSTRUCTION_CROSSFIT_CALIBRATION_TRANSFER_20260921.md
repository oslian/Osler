# Calibration Transfer Failure Localization

## Verified Comparison

Use all165target-fold combinations from the fixed33target source-only offset
experiment. Recompute every calibration offset from its original checksum-bound
calibration rows and verify patient disjointness with outer evaluation. Compute
an explicitly RETROSPECTIVE outer residual median only to diagnose whether the
source correction has the right direction/scale. The oracle is never deployed,
used to select features, or substituted into any promotion result.

Use the same equal-selected/broad-group MAE objective that defines the fitted
offset. Verify the supported retrospective median minimizes this objective
relative to both unchanged and applied offsets. At least30patients in each
group are required on each side for an interpretable comparison; unsupported
oracle is null, not zero-bias evidence.

## Results

- 165target-fold combinations retained.
- 86have sufficient calibration AND outer group support.
- 20of86require opposite-sign corrections on calibration versus outer data.
- 38of86are harmed on equal-group MAE by applying the source correction.
- 34of86are harmed on patient-average MAE.

These are descriptive counts of correlated fold-target outcomes, not86
independent statistical replications. Unsupported79combinations are not
failures or successes.

Illustrations:

| Fold / target | Source correction | Retrospective outer optimum | Outer patient MAE change |
|---|---:|---:|---:|
| 0 / albumin |+0.17065|+0.56512|-0.0915|
| 1 / albumin |+0.56512|+0.17065|+0.0487|
| 2 / albumin |+0.54326|-0.0744|+0.2063|
| 2 / total protein |-0.30603|+0.2049|+0.1108|

Fold0/1 real encoders and scalers are bitwise equivalent (prior independent
rerun evidence); their unused-inner and outer populations exchange. The
different albumin corrections are therefore not explained by different model
parameters or optimizer seeds. They expose a calibration-population difference
for the SAME fitted function. Sampling noise, case mix, measurement differences
and conditional model error are still not uniquely separated.

## Consequence For Repair

The first failing assumption of the latest repair is transport of a source
residual median: a one/few-hospital correction is not a universal model bias.
The scalar arithmetic, sign, fitting/evaluation separation and original
predictions have been checked. Changing a sign or enlarging the encoder is not
justified by this failure.

A subsequent calibration design must assess domain diversity, not only patient
counts, and demonstrate consistency across source-only calibration domains
before proposing an adjustment. It cannot reuse these retrospective optima.
The present8independent components and preassigned1-2site inner calibration
roles constrain that design; support feasibility must be checked before paying
for another full retraining. This does not prove every prediction failure is
caused by calibration or that more domains guarantee success.

Two new tests and three calibration tests pass. No weights, runtime, intervals,
promotion, data exclusions or gate changes. Report:
`artifacts/native_reconstruction_crossfit_calibration_transfer_20260921.json`.
