# Fit-Only Observation-Aligned Body Coordinates

## Question and Prior Evidence

Lossless rotations worsened finite HGB readouts despite retaining information.
PCA did not repair the gap. Actual raw-observation skip helped body readouts but
did not beat raw-only HGB. The closed kernel branch did not beat original HGB.
This test asks whether FIT observation-aligned axes make the SAME body state
easier to read, without supplying query raw measurements to the predictor.

## Frozen Design

All47 targets/all5 original source fit/selection folds. Frozen epoch10 body64,
same target/sibling-masked caches, roles, labels, weights and HGB parameters.
No outer queries, no settings chosen from selection answers. Reused source
selection is development, not a new lockbox. No runtime promotion in this test.

FIT weighted StandardScaler of body64 and raw265; Ridge(alpha=100, cholesky)
maps scaled body to scaled raw summary. No clinical target label is used to
learn axes. Raw includes latest/mean/mask/age/count; it is NOT a physiological
ground-truth hidden state. Constant fitting outputs are exactly zero.

Three329-dimensional HGB inputs, identical100-stage HGB and true target labels:
1. aligned_body: original body64 plus265 ridge-decoded observation coordinates.
2. random_axes: original body64 plus265 fixed Gaussian unit-vector projections
   of fit-standardized body; same seed20261003+target index, no outcome selection.
3. shuffled_axes: original body64 plus265 ridge outputs learned from raw rows
   cyclically reassigned within fit component, preserving all fit margins.

All three remain deterministic functions of the SAME body64, with original64
retained. No new query information or lost body information. This increases
axis availability to trees, so it is NOT a claim of exact capacity equality to
original64-dimensional HGB. The two329-dimensional controls test that confound.
Shuffling axes is not shuffling prediction labels or fully shuffling the encoder.

## Assessment and Failure Conditions

Compare aligned_body with body HGB, native neural, raw HGB, process HGB,
population, random_axes and shuffled_axes. Patient-average repeated folds first,
then equal components. Existing paired bootstrap5000 plus one-sided component/
patient test; fixed47*7 family, CI upper<0 and p<.05/(47*7), at least30patients
and3components. All comparisons required for full diagnostic screen.
No claim of independent folds or new clinical capability.

Record both fit error and selection error, not only aggregate gates. Report
observation-coordinate reconstruction MSE relative to FIT mean on observed
numeric entries separately from process fields. Neither failure of a linear
inverse nor clinical prediction failure proves irreversible encoder loss.
Axes can improve readability without improving accuracy; no repeat sweep of
ridge penalty, dimensions, seeds, targets or HGB budget after seeing answers.

Private row outputs and fitted heads stay local. Only code, tests, aggregate
reports and hashes are versioned. No weights/registry/calibrator changes.
