# Source Readout Calibration Result

Completed the frozen protocol on all available targets in six earlier V2
splits. Fit/tune/evaluation patients are disjoint within each split. Saved
prediction hashes and original tune/evaluation MAEs reproduce. Shared-state
weights remain unchanged; calibration coefficients use tune only. Both real
and process get identical offset or nonnegative affine L1 fitting.

No target significantly improves versus the original real model in all six
source evaluations under either offset or affine calibration. Consequently
none wins all six against original real, population and equally calibrated
process simultaneously. Nominal bootstrap intervals are exploratory and not
corrected for multiplicity; this is not the seven-seed promotion gate.

| Target | Original real median MAE | Offset | Affine | Affine vs original nominal gains/harms |
|---|---:|---:|---:|---:|
| Lipase |1.7030|1.7094|1.7120|1/0|
| Triglycerides |1.4929|1.4293|1.4754|0/0|
| Creatinine |1.4994|1.4810|1.4844|2/2|
| Glucose |1.2898|1.2750|1.2753|0/0|
| Bilirubin |3.1809|3.1899|3.1812|0/0|

Values are medians of six normalized current-reconstruction errors, not pooled
patient errors or physical-unit forecasts. Counts use paired intervals within
each split. Overlapping splits are not six independent external replications.

Triglyceride affine slopes are0.070,0,0,0.110,0.529,0.428. Two source calibrations
choose a constant, consistent with weak transferable ranking, not enhanced
shared-state understanding. Lipase slopes vary0.233-1.567; preserving positive
ranking alone does not establish portable amplitude correction. Creatinine
calibration improves two evaluations but harms two, despite a favorable median.

## Decision

Reject this fixed calibration procedure as a demonstrated universal repair.
Do not deploy a calibration just because pooled/median MAE looks favorable.
This does not prove all nonlinear or state-conditioned calibration impossible,
but repeating simple affine correction as an untested fix is unwarranted.

Source-support/representation generalization remains unresolved. Future
controlled changes must preserve per-target retention and patient/site
independence rather than replace the gate with favorable aggregate statistics.
No weights, runtime registry, uncertainty intervals or promotion changed.

Four calibration helper tests passed: positive-line recovery, nonnegative slope,
constant prediction and invalid values. Artifact:
`artifacts/native_reconstruction_readout_calibration_20260917.json`.
