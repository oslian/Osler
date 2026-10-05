# E44 numerical accounting amendment

The original protocol remains unchanged. Initial run19824 completed the fold0
fitting Jacobian, then exited1 at the fixed cast-decomposition check during the
fitting replay. No role parquet or completion report was saved; selection and
same-source scoring had not started. The original audit/test/protocol snapshots
are retained privately in the E44 failed_initial directory and hash-bound by
the amended audit. Earlier pre-run syntax and invalid synthetic-identity fixture
errors are recorded in graph.md, not interpreted as model results.

The float64 decomposition is orthogonal. Independently rounding each part to
float32 is not additive. The original implementation checked the sum of the
three rounded parts against the saved full direction but omitted that conversion
residual, despite the protocol requiring explicit cast/null accounting.

Retain exactly the same float32 strong and weak candidate weights, original
E43 direction, alpha, lambda, data roles and scientific comparisons. Add only:

```text
roundoff = saved_direction - (cast_strong + cast_weak + cast_null)
original_slope = strong_slope + weak_slope + null_slope + roundoff_slope
```

Keep roundoff in float64 for accounting; it is never added to either ablation
model. Report its norm and maximum directional contribution separately. The
complete slope sum must still satisfy the ORIGINAL 1e-7*(1+abs(original_slope))
tolerance. Full original-model replay and actual ablation replay tolerances
are unchanged. Do not hide a large conversion residual or call it physiology.

A synthetic cancellation test must demonstrate exact accounting while proving
the strong/weak model weights are unchanged. This fixes a diagnostic bookkeeping
error, not E43 transfer or a release model. It does not authorize parameter
search, promotion or stronger generalization claims.

## Gram Back-Projection Precision

Run90628 completed fold0 but exited1 at fold1's fixed orthogonality check, before
fold1 query scoring. Original code, tests, amendment and completed fold0 outputs
are retained in failed_gram_projection. No query MAE was used for this decision.

Fit-only diagnosis11741 recomputed the same fold1 Jacobian and compared Gram
back-projection to a direct economy SVD (SciPy LAPACK gesdd). Orthogonality error
was 2.0740969286025468e-7 versus 4.749880985481768e-11. Both methods select exactly
1952 strong and 96 weak directions at the unchanged lambda. Maximum parameter
difference is 2.612148275460413e-10; fitting-response difference <=2.3851e-11.
The independent result is native_tangent_projection_numerics_20261005.json.

Use the SVD right vectors directly for the SAME orthogonal projections, cutoff
and lambda; avoid dividing Gram-eigenvector back-projections by small eigenvalues.
Original 1e-7 orthogonality, energy, null-response and slope checks remain.
Float32 casts may differ at roundoff, so repeat all five folds, not a mixture of
solver versions. Keep the discarded partial run rather than overwriting it.

The fit-only diagnosis's original audit-source hash is now resolved against the
byte-identical failed_gram_projection/audit_tangent_spectrum.py snapshot; all
other bindings retain their original paths. Do not rewrite the historical JSON.
This is numerical implementation evidence, not a generalization improvement.
