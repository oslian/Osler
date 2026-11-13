# Patient-Excluded Baseline and Residual

Completed nested patient separation, baseline32epochs and two residual8epoch
fits. Baseline fitting/selection excludes all residual pilot patients; private
partition preserves original64fit32tune64evaluation64hospital queries exactly.
Baseline own-history uses remaining source patients, selectedepoch32. On full
eligible evaluation/hospital roles baseline MAE8.94256/10.48011 versuspersistence
10.32894/11.73923, paired CIs exclude0. No interval or promotion evidence.

Full-input residual transforms refit on baseline-fit histories only. Both state
and anchor-only corrections selectepoch1. On original64patient evaluation:
baseline9.21706 ->state9.20493, delta-.01213 CI[-.04702,.02329]. Hospital64:
baseline11.12588 ->11.12844, delta+.00256 CI[-.03239,.03710]. Neither significant.
State versus anchor-only differences also nonsignificant in both roles.

The patient exposure asymmetry was removed at native baseline fitting/selection,
but this did not establish a useful state correction in the small pilot. This
is nested holdout, not completed multi-fold cross-fitting; pretraining historical
source caveat remains. Baseline quality/selection and transforms changed too,
so no isolated causal attribution to exposure. Larger residual support remains
untested here; full-body efficacy cannot be inferred from this single target.
No serving weights, registry, calibrator or promotion changed. All direct residual
report bindings verified. Reports: `artifacts/native_nested_partition_20260913.json`,
`artifacts/map_nested_baseline_7_20260913.json`,
`artifacts/native_organ_nested_residual_20260913.json`.
