# Multiplicity Sensitivity, Not Promotion

Report: `artifacts/native_reconstruction_multiplicity_20260914.json`.
Frozen real/process/population errors, all39targets x2roles x2comparators =156
tests. No fitting or target selection. One-sided paired sign-flip tests use
99,999 Monte Carlo resamples; Bonferroni adjustment includes unsupported tests
as non-rejections. Original paired error means reproduce exactly.

This is POST-HOC sensitivity on repeatedly inspected data. Sign-flip inference
assumes exchangeable error signs under the null; it is not distribution-free
inference about arbitrary asymmetric mean-error differences. It does not replace
the original bootstrap gate or provide a new independent confirmation.

## Result

Only total_protein has adjusted p point estimates below.05 for both comparators
and both roles. Its real-vs-process adjusted p values are.04992evaluation and
.03744hospital, however, and Monte Carlo uncertainty crosses the threshold.

Using exact binomial95% intervals for the resampling exceedance proportions:

| Role | Exceedances / 99,999 | Adjusted Monte Carlo interval |
|---|---:|---:|
| Evaluation | 31 | 0.03286--0.06864 |
| Hospital | 23 | 0.02275--0.05384 |

These are simulation-error intervals, not clinical effect intervals. The
report's `retained_targets` is the point-estimate sensitivity result only;
total_protein is not a stable statistical/promotion conclusion.

The other six nominal real-over-process gains do not reject the complete
family under this sensitivity test. This does not prove no effect, invalidate
all prior bootstrap intervals, or authorize dropping their targets.

## Interpretation

Evidence is still fragile once the number of simultaneous questions is counted.
The appropriate next evidence is independent replication of the fixed complete
source-selected target set, not another threshold change or selective retest
of borderline winners. A single reused-split result is insufficient to declare
that the broad shared-state problem has been solved.

No runtime, weights, intervals or original findings were overwritten. No promotion.
