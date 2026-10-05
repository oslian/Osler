# Rotation-Invariant Probe: Complete, with a Metric Failure Located

## Fixed Comparison

All47 targets/five source folds/four arms completed:940 fits,1,797,884 source
selection occurrences,720,470 unique patient-target pairs,29,580 people. Frozen
body64 and masked raw265; true measurement labels. Fit-only RBF Nystroem128 plus
zero-epsilon LinearSVR absolute-error objective; fixed bandwidth rule, regularizer
and seed. Same raw process control and within-fit-component body permutation.
No new package, encoder retraining, external queries or source-query selection.

All940 solvers converged, in12-538iterations; no convergence warnings. Largest
rotation feature difference1.6869e-9, relative output difference5.0987e-11 across
complete fit/query arrays. Thus the same learned readout is numerically invariant
to the tested simultaneous rotation of inputs and kernel landmarks. This is not
a proof of general invariance to arbitrary rescaling, changed missingness or
nonlinear encoder information loss.

## Source Results

Normalized target-macro MAE (not physical units):

| Readout | Patient equal | Component equal |
|---|---:|---:|
| Native neural |0.882389|0.831593|
| Existing body HGB |0.857139|0.811241|
| Existing raw HGB |0.825737|0.787947|
| Kernel body |0.884742|0.834098|
| Kernel raw |0.937564|0.889135|
| Kernel process |0.939906|0.891530|
| Kernel shuffled body |0.942323|0.893573|
| Population |0.941965|0.893289|

Complete prespecified seven-comparator screen:0/47. Kernel body beats native
only for glucose, and beats existing body/raw HGB for0targets. It beats its own
raw/process/shuffled controls for17/18/22targets respectively, but these wins
must NOT be interpreted as a better overall model or complete signal retention.
Kernel body has worse patient/component means than body HGB on35/37targets,
and worse than raw HGB on46/43targets.

The ordering of body versus raw depends on the readout. In this kernel family
body is numerically better than raw for46/47patient and45/47component means;
the earlier HGB comparison favored raw. This changes the interpretation of
representation rankings, not the deployment decision: the stronger existing
readout remains better. No target promoted.

## Earliest Located Failure: Raw Euclidean Metric

After the fixed assessment, `diagnose_kernel_metric` examined only fitting data,
using exactly the1024fit-pair draws that set each bandwidth. This is an explicitly
post-result diagnostic, not a preregistered efficacy comparison or a new model.

The raw summary stores normalized latest/mean measurements beside presence,
lag in hours (missing lag49), and log counts. An isotropic Euclidean distance
mixes these units. Decomposing each positive pair's squared distance:

- Every235 target-fold combination has median lag contribution greater than90%.
- Median across those235 within-combination medians: lag99.3419%; combined
  numeric latest+mean0.4120%. These are not a global patient-weighted fraction
  and not a percentage of physiological information retained.
- Removing numeric contributions while preserving the SAME raw bandwidth changes
  pairwise kernel similarity by mean0.008963 at the median target-fold.
- Kernel body, raw and process each already lose to their corresponding HGB
  fitting MAE in232/235 comparisons. These are overlapping fits, not independent
  experiments. Solver convergence does not imply adequate predictive fitting.

This localizes a concrete defect in THIS new raw kernel probe: its distance
mostly reflects observation timing/support, not the measured numeric differences
the comparison intended to make readable. The kernel raw/process near-equality
is consistent with this measured metric dominance. It is not evidence that raw
physiology lacks predictive signal. The defect belongs to the new probe design,
not a newly discovered proven bug in the deployed JEPA.

The raw metric failure also does not explain all body-kernel harms, since body
coordinates are different. Finite landmarks, bandwidth, regularization, learned
geometry and distribution shift remain unresolved. The evidence does not permit
claiming that changing one normalizer will promote the model.

## Decision and Next Discriminating Repair

Keep candidate-only, no runtime/release/interval change. Do not repeat this
isotropic raw metric as a fair information-capacity comparison. Before further
kernel conclusions, test a fit-only scale/covariance-aware metric with identical
query roles, the same controls and explicit invariance checks; keep the current
failed metric as the comparator. This is still unexecuted, not a promised fix.
Do not tune bandwidth or transforms on the source selection answers, and do not
use already-inspected external datasets as a new lockbox.

17focused tests passed (kernel, metric decomposition, coordinate and identity
controls). Original query/reference rows and allfold metrics reproduce to1e-12;
484kernel-assessment artifact bindings verified. Private fitting heads/rows remain
ignored. No forecast, clinical, causal or calibrated-interval claim follows from
these current-reconstruction experiments.

Evidence:
- `RECONSTRUCTION_KERNEL_PROTOCOL_20261003.md`
- `artifacts/native_reconstruction_kernel_assessment_20261003.json`
- five `artifacts/native_reconstruction_kernel_foldN_20261003.json` reports
- `artifacts/native_reconstruction_kernel_metric_20261003.json`
- `scripts/train_kernel_probe.py`, `scripts/score_kernel_probe.py`
- `scripts/diagnose_kernel_metric.py`

Implementation follows maintained library components:
https://scikit-learn.org/1.7/modules/generated/sklearn.kernel_approximation.Nystroem.html
https://scikit-learn.org/1.7/modules/generated/sklearn.svm.LinearSVR.html
