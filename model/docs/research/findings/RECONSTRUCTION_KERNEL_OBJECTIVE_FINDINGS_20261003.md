# Kernel Fit Failure: Shrinkage Exists, but It Explains Little of the Gap

## Scope

Fit-only diagnosis of all47 targets/all5 source folds, original and scaled
body/raw/process maps:1,410 saved map/head combinations. Reconstructed original
fit predictions/MAE, verified original artifact hashes and fitting roles. No
new head, no query/outer inference, no promotion or runtime changes. This is
current-value reconstruction, not future forecast or physiological causality.

The preceding experiment fixed raw lag dominance and reduced error, but all
three scaled families still lost their corresponding HGB FITTING error in
232/235 cases. This diagnosis asks where that earlier-than-transfer gap arises.
It differs from the September20 linear-readout bottom16 ablation: today's
objects are nonlinear128-landmark RBF features and regularized absolute-error
SVR coefficients, not the native64dim body linear correction.

## Not a Constant Representation

Weighted centered feature spectrum, median across235 overlapping fit problems:

| Readout | Entropy effective rank | Rank for95% variance | First-mode share |
|---|---:|---:|---:|
| Old body kernel |27.16|54|18.71%|
| Scaled body kernel |31.39|58|16.06%|
| Old raw kernel |24.44|49|23.74%|
| Scaled raw kernel |33.91|58|17.99%|

These features are not constant or one-dimensional. This does NOT prove they
retain all relevant information, have enough useful modes, or correspond to
physiological mechanisms. Effective rank is scale/distribution-dependent.

## Regularization Compresses Predictions

Use the actual LinearSVR objective, including penalized synthetic intercept:
0.5*coefficient_norm_squared + C*sum(weight*absolute_error). External fitting
median offset remains fixed. Along the current coefficient/intercept direction,
derive the exact nonnegative unregularized L1 gain using a weighted median.
This uses FIT truth, never unseen outcomes or a deployable routing feature.

| Scaled source | Prediction/truth weighted SD | Optimal fit-only L1 gain | Fit MAE reduction |
|---|---:|---:|---:|
| Body |0.1761|1.3666|0.6900%|
| Raw |0.1097|1.7865|1.3114%|
| Process |0.0768|1.5587|0.3374%|

All entries are separate medians, not a single pooled model. Lower predictive
dispersion is not itself a defect: conditional-median prediction is normally
less variable than noisy outcomes. Nevertheless, body232/235 and raw235/235
fit problems prefer gain>1.05 for unregularized training L1 loss. All235 scaled
body and raw problems incur WORSE penalized objective at that fit-L1 gain.
The regularizer therefore creates an observable amplitude/loss tradeoff.

Numerically minimizing the SAME regularized objective along this ray improves
it by at most3.883e-7 for scaled body,5.227e-7 for scaled raw (relative to
max(1,absolute objective)). No scaled problem exceeds1e-6. Original unscaled
raw has one case at1.725e-6. There is no large error along this tested direction
that can be attributed to stopping before the ray optimum. This is NOT a full
multivariate KKT/duality certificate; other coefficient directions remain open.

## But Amplitude Alone Cannot Repair the Readout

Compare same-row, same bounded-weight fitting errors with original HGB. Even
oracle TRAINING gains leave232/235 body and232/235 raw problems worse than HGB.
Among initially positive gaps, median fraction recovered is only:

- Scaled body:5.043%.
- Scaled raw:7.063%.
- Scaled process:3.898%.

For BUN, median body/raw gains1.5297/2.1636 reduce fit MAE only1.452%/2.897%.
For creatinine, gains1.6475/2.0961 reduce fit MAE1.071%/1.591%.
These are retrospective training diagnostics, not proposed patient-specific
corrections and not expected improvements on another hospital.

Thus reduced amplitude is REAL but insufficient. Most of the measured fit gap
remains even after the best nonnegative gain on the current learned direction.
It cannot all be explained by external domain shift, strict acceptance gates,
or a single missing rescale. Finite feature approximation, conditional map
expressivity, and the learned coefficient directions remain unresolved. We have
not established a unique root cause in the original JEPA encoder.

## Decision

No blanket gain, weaker deployment gate or promoted kernel. Keep strong HGB and
native references. If investigating this readout further, the next discriminating
intervention must change coefficient directions while holding the SAME feature
map fixed: a prespecified fit-internal regularization comparison can distinguish
penalty-limited fitting from a weak finite map. It must retain held-component
selection and controls; absence of improvement should end this kernel repair
branch, not justify a query-tuned parameter sweep. Not yet executed.

20focused tests pass: exact signed weighted-L1 gain, known regularized optimum,
detectable ray suboptimality, zero predictions, spectrum rotation, constant
truth handling, valid gap denominators, plus previous scaling/kernel tests.
Private features and heads remain local. No actual patient predictions deployed,
forecast/interval claims or new external validation.

Evidence:
- RECONSTRUCTION_KERNEL_OBJECTIVE_PROTOCOL_20261003.md
- scripts/diagnose_kernel_objective.py and scripts/summarize_kernel_ray_gap.py
- artifacts/native_reconstruction_kernel_objective_20261003.json
- artifacts/native_reconstruction_kernel_ray_gap_20261003.json
