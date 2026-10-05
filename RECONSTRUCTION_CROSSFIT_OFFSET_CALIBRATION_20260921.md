# Independent Offset Calibration: Completed

## Actual Repair Tested

Frozen fresh encoders, all33targets/all5folds. Predict the preassigned inner
encoder_tune patients, who were excluded from the corresponding encoder fit
and outer test. Estimate one source-calibration equal-group residual median
per target/fold. Apply it to every outer patient without observing group/site/
outcome. No variant selection. All247611outer pairs retained,247610eligible.

Calibration pair counts per fold:57871,71066,71066,71066,71066. These counts are
NOT independent or unique patients: inner roles repeat across folds. Fold0 has
five targets below the required30patients in each group; their corrections
remain zero with explicit unsupported status. Other folds support all33.
Calibration domains are thin (the preflight's1-2sites); patient count alone
does not imply representative hospital coverage.

## Results

Macro normalized MAE:

| Scope | Unchanged | Corrected |
|---|---:|---:|
| All |1.673005|1.673893|
| Selected |1.874418|1.886968|
| Broad |1.663383|1.664499|

Numerical patient-mean improvements:24/33all,14/33selected,24/33broad. No target
passes the incremental descriptive screen against unchanged, process AND
population controls in any scope. This is not a retraction of the four targets
that passed the earlier, less extensive descriptive comparison; the current
question additionally requires a benefit over the unchanged real model.

| Target | Selected MAE change | Broad MAE change |
|---|---:|---:|
| Creatinine |-0.015777|-0.020232|
| BUN |+0.002095|-0.017417|
| Glucose |+0.000227|-0.020121|
| Albumin |+0.062509|+0.055155|
| Hematocrit |-0.000250|-0.002400|
| Total protein |+0.004102|+0.031341|

Positive means harm. Creatinine selected patient MAE improves1.292164->1.276387,
but its equal-component mean difference is+0.015091. Calibration can improve
the patient-weighted average without fixing hospital robustness. Albumin and
total protein regression also prevent a safe all-target default change.

## Interpretation

The retrospective creatinine offset oracle was a useful diagnostic but not
a deployable correction. Independent source calibration helps some targets,
yet the desired corrections do not transfer uniformly between groups and
hospital components. A fixed intercept is therefore not a complete root-cause
repair. The result does not uniquely attribute failure to biology, encoder,
domain composition or irreducible information limits.

Do not sweep intercept shrinkage on these outer results, pick the24improving
targets, or ship hospital-specific corrections. Keep the unchanged release.
This completed source-calibration experiment is a negative control against
the claim that a simple bias fix would solve the latest cross-fit failures.
Further work needs source-only evidence for conditional corrections or a
training-objective mismatch, not another posthoc scalar adjustment.

Three tests pass: calibration group replication invariance, insufficient
support leaves zero correction, and correction sign. All inference completed
normally and no training was interrupted. Source roles, checksums and patient
disjointness validated. No weights, runtime, interval or promotion change.
Report: `artifacts/native_reconstruction_crossfit_offset_calibration_20260921.json`.
