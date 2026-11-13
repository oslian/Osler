# Rotation-Invariant Nonlinear Readability

Fixed all47/five source folds, same frozen body64 and masked raw265, same fitting
and held-component selection queries. Prior lossless rotation showed coordinate
bias in finite HGB. This tests a different nonlinear readout with an explicit
rotation-invariance check, not another coordinate search or new body encoder.

Use installed scikit-learn Nystroem(RBF,128fit-only landmarks,seed19), followed
by LinearSVR(loss=epsilon_insensitive,epsilon=0,C=100/n_fit,tol=1e-6,
max_iter=50000,dual=True,fit_intercept=True,random_state19). This is regularized
absolute-error fitting in a finite kernel feature space, not squared-error ridge.
Center inputs on fit-only weighted means, NO per-feature scaling/whitening.
Gamma is inverse median positive squared distance from1024 deterministic pairs
sampled from fitting rows using bounded component weights (seed20261003+target
index); use gamma1 only for identical fitting rows. Offset labels by the fitting
weighted median, restoring it at inference. No hyperparameter/target selection.

Four arms: kernel_body, kernel_raw, kernel_process(raw first106values zero),
kernel_shuffle. The last shuffles body kernel features within fit components,
retaining the SAME body map/landmarks, truth and weights; query uses own features.
Record singleton donors. Same128feature maximum and optimization settings,
not a claim that original body/raw inputs have equal information or dimension.

For body/raw/process, rotate centered input AND landmarks by a fixed orthogonal
matrix, reuse the fitted landmark normalization, and check full fit/query kernel
features and outputs against originals (absolute feature tolerance1e-6, output
relative-to-max-magnitude tolerance1e-6). This tests the same learned function,
not training on one coordinate system and evaluating in another. All fitting
convergence warnings and iteration counts retained; no promotion interpretation
if any solver does not converge. Do not silently increase budget/restart on data.

Primary kernel_body comparisons: native, body_probe, raw_probe, population,
kernel_raw, kernel_process, kernel_shuffle. Family47*7; patientfold-first then
component-equal MAE,>=30patients>=3components,pairedbootstrap5000/95%upper<0,
one-sided t p<.05/family and mean<0 at BOTH levels, alltargets reported. No outer
data. No revised gate, routing, numeric clipping or posthoc winning subset.

Failure does not prove missing information: fixed bandwidth, landmark budget,
Euclidean metric, regularization and input semantics may limit this learner.
Raw includes medical-process inputs; success would not establish physiology-only
signal or forecasts. Source roles reused and folds overlap; bootstrap conditional
on models. This screen cannot grant runtime/interval/causal promotion. Private
patient rows and diagnostic models stay ignored; bind all scripts and artifacts.

Library patterns, no new package:
https://scikit-learn.org/1.7/modules/generated/sklearn.kernel_approximation.Nystroem.html
https://scikit-learn.org/1.7/modules/generated/sklearn.svm.LinearSVR.html
