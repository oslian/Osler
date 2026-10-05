# Signed Distribution Follow-up: Fixed Before Execution

This follows the failed two-target prospective-change experiment. It is a new
development experiment, not a rerun of its frozen protocol or a new lockbox.
Targets remain glucose@3h and MAP@3h. Release weights and runtime stay unchanged.

## Fixed Differences

1. Fit 19 signed-delta quantiles (.05, .10, ..., .95) on the same frozen JEPA
   features and seven history features. Reuse scikit-learn HGB with the prior
   fixed capacity (80 iterations, 15 leaves, min leaf 40, L2 1, learning rate .05,
   seed 20260906). There is still no restriction to shrinking an expert's delta.
   Direct, shuffled-history, and history-only have identical quantile counts and
   estimator settings. This adds distribution resolution, not a larger encoder.
2. Calculate P(delta<=-threshold) + P(delta>=threshold) from the piecewise-linear
   quantile function, with constant endpoint tails. The tails are approximate,
   not evidence of bounded biological changes. Apply source-only Platt calibration
   for alert probabilities. After Platt, calibrated risk is not claimed to be
   exactly the raw quantile distribution's tail mass.
3. Compare two predeclared CQR methods: pooled additive expansion, and expansion
   normalized by max(raw 90% quantile width, research change threshold). All
   calibration patients contribute one score each; minimum remains 100. This
   changes the calibration estimand from risk-bin to marginal coverage, NOT the
   held-out [87%,93%] overall/risk-bin/care-unit assessment requirements. No method
   is selected after viewing NWICU. Report both, including widening and failures.
4. Evaluate the old independently classified risk and old point readout as an
   explicit comparator. For interval comparisons attach the same new CQR methods
   to its raw quantiles as an ablation, not the unavailable old glucose intervals.
   Keep release intervals untouched. Linear/persistence experimental intervals
   use source-calibrated residual scores, not fabricated release distributions.

Use exactly the prior eICU role masks, anchor hashes/caps, target-specific future
tolerances and current age <=1h; patient_test and hospital_test remain untouched
by fitting. NWICU uses the prior 500-patient/target hashes and chronological replay.
The old candidate artifact must match the frozen release. Compare q05/q50/q95
raw estimators against their prior counterparts on validation inputs; report
differences. Sorting 19 quantiles may move the median relative to sorting three,
so any tiny point difference is not evidence of learning new physiology.

Risk cutoffs are fixed from eICU validation at <=5% empirical FPR (strict >).
Common predicted-risk groups retain the prior head's source-derived boundary,
so interval methods cannot redefine the evaluated groups to look better.
Use patient-paired bootstrap, same seven evaluation resamples (NOT training
seeds), held hospital and care-unit slices, and late-stay chronological replay.
At equal FPR report paired-bootstrap recall differences versus release, history
and shuffled controls; thresholds on held-out labels are ranking diagnostics only.
Report mean pinball loss on the 19-quantile grid as a distribution score, not an
exact CRPS for a fully known distribution. Report signed delta and interval scores.

All new heads, probability calibration and CQR radii are serialized and hashed
BEFORE reading NWICU outcomes. NWICU was already inspected in prior work; no
pristine external-validation claim. Frozen encoder training exposure remains
unverified. Original promotion gates, new independent data, full training-seed
replications and true onset labels are not supplied by this experiment. Late
measurement lead time does not prove prediction before biological onset.

No post-result threshold changes, no runtime promotion, no causal/clinical claim,
no next-measurement-time model. Save aggregate reports and candidate weights only.

Methods: [scikit-learn quantile regression](https://scikit-learn.org/1.5/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
and [Romano et al. CQR](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).
