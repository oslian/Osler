# MAE-Aligned Source-Site Repair Protocol

Frozen bounded7 state, all33 targets, same source data/linear capacity. Fit
weighted MAE plus0.1 times L1 norm of standardized coefficient departure
(including intercept) from original. Fixed0.1, no hyperparameter search.
Matched arms differ ONLY by per-group/site empirical MAE <= original.
Both differ from preceding SmoothL1/L2 study; do not attribute all differences
to constraints. Include original as baseline.

Use installed scipy HiGHS sparse linear programming with absolute-residual
and absolute-departure epigraphs. Source-only center/scale and bounded weights.
Reject solver failure, nonfinite coefficients, objective mismatch>1e-6,
objective worse than feasible original>1e-7, or site violation>1e-7.
Save all coefficients before selection scoring. Existing incremental gate
requires patient/site improvement against original AND matched unconstrained.
No deployment selection, fresh lockbox, placebo/interval promotion claim.
Do not tune alpha from these outcomes. This experiment repairs metric
alignment, not a promise of universal cross-hospital physiological transfer.
