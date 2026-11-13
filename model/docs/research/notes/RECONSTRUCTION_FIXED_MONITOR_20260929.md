# Fixed Source Monitoring Contract

The eight-epoch experiment is complete and immutable. Its old online losses
cannot reconstruct a fixed-patient learning curve. This tool establishes the
epoch8baseline for a FUTURE monitored continuation; no continuation run or
best-epoch policy is claimed complete here.

Selection uses only patient/target identity SHA256 ordering, at most32people
per target/component. All47targets and smallgroups retained. Inputs are strictly
the existing source-selection role, never fit/calibration/test. Counts per fold:
6862,6862,7167,7167,7167 patient-target pairs. Overlap acrossfolds remains and
does not create independent replication. Private keys bind future comparisons.

Current baseline uses already verified frozen predictions from coverage
real/process and replacement real/process, plus population center. Metrics:
target patient-equal MAE, component-equal MAE and SmoothL1 on the same examples.
Future checkpoint inference must reproduce these keys and use unchanged fit
scalers, masks and times; incomplete/duplicate/nonfinite outputs fail closed.
Do not add monitor queries to optimizer draws. Monitor after each saved epoch,
not during changing online weights; keep each checkpoint and monitor report.

This is a source diagnostic cohort, not an untouched validation set, runtime
gate, or justification to stop at a favorable epoch. Predeclare continuation
budget and comparison before updates; preserve original epoch8control and all
targets. Full frozen outer comparison remains required for capability claims.

Reproduce:`python -m scripts.fixed_reconstruction_monitor`.
Private identities:`data/research_recovery/additional_development_domains_20260921/fixed_monitor_20260929/`.
Aggregate:`artifacts/native_reconstruction_fixed_monitor_epoch8_20260929.json`.
Four unit tests cover ordering independence, role isolation, missing predictions
and weighting of smallgroups. No training/runtime/promotion changed.
