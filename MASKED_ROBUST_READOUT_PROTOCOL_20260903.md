# Robust Frozen-Body Readouts: Pre-run Protocol

Written before fitting or inspecting this experiment's external scores.
Exploratory same-anchor reconstruction; no runtime promotion.

## Question

Can a different statistical readout extract more transferable current-value
information from the same shared JEPA body state without changing that state
or its existing forecasts? This does not test a new parallel body encoder.

## Fixed Comparisons

Reuse the previous run's source cohort, 1,736 fit / 1,737 tune patients and
1,288 patients from 55 intervention-heldout hospitals. Validate their hashes,
source split and reference predictions. All 15 numeric targets, fresh truth
<=1h, legal inputs <=6h, complete target/sibling/history masking. No hospital,
care-unit, actual hidden target, treatment or future-label inputs added.

Readouts of the SAME frozen 64-dimensional context:

- Median regression: scikit-learn QuantileRegressor, quantile 0.5, L1 alpha
  0.01 or 0.1, HiGHS solver.
- Robust linear regression: HuberRegressor, epsilon 1.35, L2 alpha 0.1/1/10,
  max_iter 2,000, tolerance 1e-5.
- Small nonlinear tree readout: HistGradientBoostingRegressor absolute-error
  loss, 128 iterations, learning rate 0.05, 7 or 15 leaves, minimum leaf 50,
  L2 10, no internal early stopping or validation split.

Use the existing scikit-learn 1.7.2, no new dependency. Source-only StandardScaler
inside each pipeline. Seven patient bootstrap bags (existing seeds); real and
placebo use identical resampled recipient rows and labels. Placebo contexts
are independently deranged across distinct patients BEFORE resampling.
Each real/placebo ensemble has seven models of the same configured architecture.
Fit convergence warnings invalidate a configuration rather than being ignored.

Select real and placebo hyperparameters independently using the existing eICU
tuning objective: half mean normalized MAE + half worst supported care-unit MAE.
Also retain the placebo with exactly the real-selected complexity; the new arm
must pass against BOTH controls. No external selection of seed, method or alpha.

A single source-selected portfolio is prespecified: choose among these three
families and the previous latent ridge, fixed-32 neural ensemble and
source-selected-duration neural ensemble, using only the same tuning objective.
Each source-selected arm inherits its own full diagnostic, never the best
external target result. Report every arm including failures.

## Outcomes

Use the unchanged population / three wrong-patient / independently trained
placebo / seven patient split / all-supported-care-unit diagnostics. Also
compare directly with fixed-32 and selected-duration neural reference heads.
Report fit-defined low/middle/high truth strata (10th/90th percentile) MAE and
bias. Strata are evaluation only, not model inputs, abstention, or routing;
no external outcomes tune the boundaries. Extreme values may be real disease,
not noise. Robust loss succeeding on averages is not proof of safe tail behavior.

All models are frozen before external evaluation. MIMIC and NWICU have already
been inspected; no new lockbox claim. The source hospital holdout does not
guarantee the historical encoder never saw those patients. Seven evaluation
splits overlap and many nominal comparisons are not multiplicity-adjusted.
No conformal interval or future-cell promotion follows from a reconstruction
pass. Existing weights and normal forecasts remain unchanged. No raw patient
rows committed, no next-measurement prediction, no causal or clinical claims.

References (existing maintained implementations):
[QuantileRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.QuantileRegressor.html),
[HuberRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.HuberRegressor.html),
[HistGradientBoostingRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
