# Frozen Kernel Fit Diagnosis

Inspect all47 targets/five source folds, original and scaled body/raw/process
maps. No new heads, no selection or external rows, no serving change. Distinct
from September20's linear body bottom16 correction ablation: these are current
128-dimensional nonlinear kernel features and epsilon-zero regularized L1 heads.

For each saved model reconstruct original fit predictions and verify recorded
fit MAE. Report weighted centered feature covariance eigenvalues, entropy
effective rank,95% variance rank, first-mode share and feature trace. Report
weighted output/label standard deviation, patient IQR ratio and fit correlations.
Neither low output variance nor low effective rank alone proves collapse.

Recover LinearSVR's actual objective including its penalized synthetic intercept:
0.5*(||coef||^2+(intercept/intercept_scaling)^2)
+ C*sum(sample_weight*abs(prediction-(truth-offset))).
Compare with the zero-coefficient weighted-median predictor. Along the frozen
coefficient/intercept RAY, analytically find the nonnegative MAE-minimizing gain
as a weighted median of residual_target/prediction_delta with weights
sample_weight*abs(prediction_delta). Zero deltas contribute constant loss.
Then numerically minimize the regularized objective on that same ray, retaining
gain0/gain1/exact LAD gain as candidates. Report fit loss decrease and penalty
increase separately. This is an exploratory fitting diagnostic, NOT a selected
new deployable gain. It cannot prove full multivariate solver optimality or
show that removing regularization improves unseen patients.

All identities, original weights, cache and artifact bindings checked. No
posthoc clipping/removal, no masks changed, no bootstrap promotion. Model arrays
and rows stay private; commit only aggregates. The unknowns remaining after this
diagnosis will determine any subsequent prospectively fixed source experiment.
