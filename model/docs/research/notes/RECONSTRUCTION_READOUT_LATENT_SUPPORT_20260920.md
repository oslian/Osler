# Current Readout Latent Support Diagnostic

## Question And Fixed Method

Does current anchored-head transfer regression occur only outside source-state
support? Use all 33 trained targets and both existing source groups. For each
target/group, standardize its 64-dimensional frozen body features using fit
patients only, estimate covariance with sklearn LedoitWolf, and fix a squared
Mahalanobis-distance threshold at the fitting distribution's empirical 99th
percentile. Score encoder-selection patients without changing the threshold,
weights, heads, inputs, labels or deployment behavior.

Full cached feature digest and parent dependencies verified. Every original
versus anchored patient-average error difference reproduces the parent report.
This report uses patient averages within strata, unlike the preceding
group/site-equal transfer audit. Do not compare their numbers as if they had
identical weighting.

## Results

Median outside fraction across targets is 9.52% for selected patients and
1.50% for broad patients. Of 17 target/group pairs with overall mean regression,
14 also regress inside the ellipsoid with at least 30 inside patients.
These are numerical/descriptive counts, not significance or promotion counts.

| Target/group | Inside patients | Outside fraction | Inside mean error change |
|---|---:|---:|---:|
| Glucose/selected | 85 | 5.56% | +0.00794 |
| Glucose/broad | 3084 | 1.38% | +0.01849 |
| PTT/selected | 53 | 1.85% | +0.07299 |
| PTT/broad | 1317 | 0.75% | +0.01846 |
| Troponin I/broad | 782 | 3.58% | +0.17252 |

Positive means anchored refit is worse than the original head, in normalized
target units. No original clinical-unit interpretation is intended.

## Interpretation

Simply rejecting states outside THIS fit-only ellipsoid would leave substantial
inside-support regression. Thus this diagnostic does not support a distance-only
abstention rule as the complete repair. It does not disprove other kinds of
distribution shift: global ellipsoids can contain low-density regions, and
matching distance does not imply matching physiological states, labels or
measurement processes. The empirical in-sample threshold is not calibrated
99% novelty coverage; small-sample covariance/threshold uncertainty remains.

Together with nested CV, this narrows the next question to whether the trained
representation/readout relation remains valid on independent hospitals, not
merely whether those hospitals produce extreme latent vectors. It does NOT
prove that a new architecture, more data, or a particular adaptation will fix
the relation. Do not choose a new distance quantile using these outcomes.

Two tests pass: adding extreme query data cannot alter the fitted threshold or
existing query distances; empty strata report no mean. Audit exit 0. Aggregate
report only; no patient records added to version control. No runtime, weights,
calibration, interval or promotion changes. Current masked reconstruction only.

Evidence: `artifacts/native_reconstruction_readout_latent_support_20260920.json`.
