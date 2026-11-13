# Source-Site Constrained Readout Protocol

Fixed before this run: current bounded7 body, all 33 trained targets, exact
cached source-fit and encoder-selection pairs. No new features or capacity.
Compare original, fixed strength1 anchored SmoothL1, and the same objective
with per-(source group,hospital) mean SmoothL1 no higher than original.
Use all source sites, including small ones; this may be conservative and is
not a population guarantee. Source-only standardization and coefficient fitting.

Use scipy SLSQP, analytic gradients, max500, ftol1e-10; accept only solver
success, finite coefficients and minimum constraint slack >=-1e-8. Otherwise
keep original exactly and report failure explicitly. No penalty/threshold sweep.
Save all coefficients before scoring selection data. Retain existing patient
and site incremental gate, requiring constrained to beat original and matched
unconstrained; no replacement of full placebo/interval/runtime promotion.

This differs from older robust losses/outlier handling and protected body-
dynamics gradient probes: constraints apply to the exact fixed-state linear
readout empirical risk, with the body frozen. Encoder and selection cohorts
are reused research data, not a fresh lockbox. No clinical/causal claim.
