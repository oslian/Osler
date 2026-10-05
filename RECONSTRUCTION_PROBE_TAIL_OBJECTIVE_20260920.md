# Probe Tail And Objective Diagnosis

Fixed current raw/body probe, no refitting or exclusion. Decompose its body-
ridge versus original macro regression by target. Fit two constant controls
using only source fitting labels and the same bounded-site/equal-group weights:
the weighted mean (squared-error optimum) and weighted median (MAE optimum).
Define a descriptive tail using the unweighted99th percentile of source
absolute distance from that median. Do not clip, delete, route or select using
these labels. Tail loss share uses squared deviation from the source mean.

Troponin I and CPK account for77.92%selected/80.40%broad of the summed positive
per-target ridge regressions. Denominator excludes improvements; it is not a
percentage of all prediction errors or patients. The top-two ranking is
post-hoc diagnosis, not a reduced target catalog.

CPK's tail is1.507%of weighted source mass yet contributes98.60%of source
squared loss. Its weighted mean6.696 differs strongly from median-.0063 in
normalized units. Mean/median constant MAE is10.920/7.255selected and7.373/
3.137broad, compared with body-ridge21.756/18.245. Troponin I also has large
mean/median separation; its specified tail contributes24.93%, so do not
generalize the CPK98.6%figure to all heavy-tailed targets.

Across33targets, mean/median constant macro MAE is2.29245/1.97093selected and
2.20033/1.68342broad. Original1.79798/1.55977 remains better on average; constant
controls are diagnostics, not a proposed replacement. PTT selected is an
explicit counterexample to universal median superiority: mean2.13792 vs
median2.61865, while body-ridge1.90858 improves both constants.

## Interpretation

The very poor ridge macro must not be read as uniform shared-state collapse.
It is dominated by a few skewed targets, and even constant-estimator behavior
depends strongly on loss preference. This makes the ridge probe an inadequate
sole test of information retention. It does not prove that heavy tails alone
caused each fitted ridge error: feature-target relations, regularization and
transfer remain possible. Raw-PCA information loss also remains outside this
constant-control diagnostic. A robust raw-versus-body comparison would need
the same objective in both arms and all targets retained, not outcome-selected
tail deletion. Current body-only robust-head refits already exist and failed
full transfer; they cannot be silently called untested.

Two tests pass for weighted median and tail/zero-loss cases. Exact body cache
digest and parent dependencies verified. Audit exit0. No weights, runtime,
intervals, registry, prediction catalog or promotion changed. Current masked
reconstruction only; no causal or clinical claim.

Evidence: artifacts/native_reconstruction_probe_tail_objective_20260920.json.
