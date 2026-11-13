# Source-Site Constrained Readout Results

Protocol fixed before execution in RECONSTRUCTION_SITE_CONSTRAINED_PROTOCOL_20260920.md.
All33 trained targets, current frozen bounded7 body, same 64-dimensional linear
heads, fixed anchoring strength1. Compare original, unconstrained anchored,
and identically anchored with each source-group/hospital mean SmoothL1 no worse
than original. No per-target choice after looking at selection outcomes.

All33 SLSQP solves succeed (12-19 iterations); 15-22 constraints per target.
Worst numerical slack magnitude below 7e-11, within prespecified1e-8 tolerance.
All source weighted training losses decrease. This is empirical optimization,
not a hospital-population safety guarantee. All coefficients saved before
held-source scoring, with hashes and complete original cache verification.

| Source-selection macro MAE | Original | Unconstrained strength1 | Constrained strength1 |
|---|---:|---:|---:|
| Selected | 1.797984335 | 1.789483853 | 1.788582722 |
| Broad | 1.559770689 | 1.551793029 | 1.550220734 |

Constrained improves original macro by about0.52%selected/0.61%broad, and the
matched unconstrained arm by about0.05%/0.10%. These are target-equal averages
of patient errors in normalized units, not pooled clinical-unit performance.
21/33targets improve numerically in bothgroups versus original;15/33 do so
versus matched unconstrained. None pass the fixed incremental patient/site
gate against BOTH references. No promotion or selected replacement.

## Examples And Localization

- Glucose: patient means improve in bothgroups versus original, but selected
  equal-site mean worsens (+.008825); broad site evidence remains insufficient.
- BUN: broad improves original .893612->.841445 and passes that comparison,
  but is worse than matched unconstrained .839286. Not a constraint-specific win.
- PTT: constraints reduce some unconstrained regression, but selected/broad
  2.165704/1.387703 remain worse than original2.090042/1.375589.

Thus enforcing source-site risk retention is feasible and helps aggregate
transfer modestly. It does not solve the remaining conditional relation or
representation transfer gap. The gains over original must not all be credited
to site constraints: most are already present with fixed-strength anchoring.
PTT shows that even meeting every observed source-site constraint does not
ensure improvement on unseen source hospitals. SmoothL1 constraints and MAE
evaluation also differ; neither measure is interchangeable with clinical safety.

Prior robust-readout studies used other encoders/cohorts and robust losses;
protected body-dynamics probes changed another parameter block. This experiment
specifically adds exact convex empirical constraints to the current frozen-body
head, not a new body model or additional nonlinear capacity.

Two tests pass: analytic constraint Jacobian agrees with finite differences,
and a synthetic perfect site cannot be sacrificed to improve another site.
Audit exit0, diff check passes. Current masked reconstruction only. No runtime
weights, registry, intervals, causal permissions or formal promotion changed.

Evidence: artifacts/native_reconstruction_site_constrained_readout_20260920.json.
