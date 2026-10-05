# Numerical Continuation And Incomplete-Comparison Audit

The prior160incomplete target-folds were not all optimization failures:
38had both insufficient support and no seen-domain exposure;72support only;
43no exposure only;4no query;1nonconvergence only;2support plus nonconvergence.
Thus only3targetfolds contained five nonconverged seen-component fits.

## Fixed Numerical Repair

Allfive original300-iteration failures were reproduced (iterations and fitted
scale checked). Continued the SAME Huber model via warm_start for up to2700
additional iterations, without changing data, features, alpha, epsilon or tol.
Allfive emitted no convergence warning after continuation. This is solver
termination, not proof of numerical uniqueness or physiological correctness.

| Target/component | Additional iterations | Query MAE original -> continued | Unchanged model MAE |
|---|---:|---:|---:|
| O2sat/328 |136|2.67954 ->2.67963|2.41894|
| Triglycerides/85 |139|60.40152 ->60.33164|49.25982|
| Uric acid/146 |58|4.07824 ->4.07870|4.55571|
| Uric acid/328 |81|1.98481 ->1.98488|1.68454|
| Uric acid/353 |572|2.34087 ->2.34708|2.16100|

Prediction shifts are small; four remain worse than unchanged. Optimization
budget was a real completion obstacle for these fits, not an explanation of
the general accuracy failure. Fitted residual scales remain very small; no
arbitrary scale floor was introduced to manufacture convergence or accuracy.

## Completed Paired Diagnosis

Recomputed unaffected process/unseen comparisons and applied only the five
numerical continuations. All75previously complete target-fold records remain
exactly equal. O2satfold0 becomes complete: total76;159remain incomplete due
to data support/exposure/query absence. Numerical repair cannot create support.

Within76:17both seen/unseen better than unchanged,8seen-only,4unseen-only,
47neither. Seen beats unseen in61. These descriptive means still do not
establish formal promotion or a sole root cause. The original75comparison
artifact is preserved, not silently overwritten.

The new completion utility initially required the untouched unseen arm to be
fitted even in known support-limited groups. It aborted before writing a report;
corrected to preserve those groups as incomplete rather than treating numerical
repair as restored data support. This new utility assertion is not a historical
model defect.

One new numerical continuation test passes. No external data, encoder update,
runtime, intervals or promotion. Same current masked reconstruction task.
Artifacts: `native_reconstruction_robust_convergence_20260925.json` and
`native_reconstruction_seen_domain_completed_20260925.json` in `artifacts/`.
