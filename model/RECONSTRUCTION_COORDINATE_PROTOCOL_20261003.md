# Lossless Coordinate Readability Audit

The frozen body/raw HGB gap is not by itself evidence of information loss. HGB
uses feature-wise binning/splits, so distributed coordinates and named raw
measurements need not have equal accessibility for the finite learner. Test this
confound before another encoder redesign. Prior PCA64+ridge reduced dimension;
this intervention preserves ALL dimensions and uses the current absolute-error
HGB, all47 targets, five source folds, identical patient/query/mask contracts.

Keep original body64/raw265/native/process/population controls. Four new heads:
body_pca, body_rotation, raw_pca, raw_rotation. Each uses the same existing fixed
100-iteration HGB. PCA is the complete orthonormal eigenbasis of the fit-only
bounded-component-weight covariance, with NO truncation, whitening or rescaling.
Random rotation uses scipy.stats.ortho_group with one fixed seed (20261003 +
variable index), no outcome selection. Both arms subtract the SAME fit-only mean.
Retain null-variance directions, validate orthogonality and inverse reconstruction
on fit AND query rows. Queries never determine transformation parameters.

Fit roles may already have been seen by the frozen encoder; queries remain
patient/component disjoint. No outer queries, future labels, new observations,
outcome-defined routing or patient-state null claim. Randomly rotating coordinates
is NOT shuffling patients. It preserves all information and does not remove
medical-process information. Body/raw dimensions and semantics remain different.

Primary attempted repair: body_pca must beat body_probe, native, raw_probe,
process_summary, population, and body_rotation at patient AND component levels.
Additional diagnostic contrasts: raw_pca-raw_probe, raw_rotation-raw_probe,
body_rotation-body_probe. Family=47*9, >=30people and >=3components, original
paired bootstrap5000/95% upper<0, one-sided t p<.05/family and mean<0. Report
both positive and negative effects, all47 targets, no selection or deployment.
Patient losses average folds first. Overlapping folds and conditional bootstrap
do not constitute independent model refits. This source-only screen cannot
authorize runtime promotion even if passed.

If lossless raw rotation worsens errors, a coordinate/finite-readout confound is
demonstrated without data loss, not proof that all encoder gaps have that cause.
If body PCA improves, accessibility is partially repaired without extra input.
If neither repairs the gap, do not infer that information is absent. No arbitrary
new rotation seeds after inspecting outcomes. Bind scripts, input caches, models,
predictions and artifacts. Keep patient-level rows private; runtime unchanged.

Implementation uses installed SciPy and scikit-learn (no dependency addition):
- https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ortho_group.html
