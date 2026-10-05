# E36: saved bridge parameter direction, not output interpolation

## Fixed question and scope

E35 shows that a trained output correction often points away from truth. It
does not show whether the recurrent parameter direction is harmful at zero.
Use the E34 immediate arm, the prespecified primary anatomical-writeback arm,
all five folds, all fit/selection/same-source queries and all 53 inventory
targets. This is 20,480 queries, not a selection of near-passing targets.
Delayed/body arms were covered in E35 but are NOT replayed here. Conclusions
are restricted to immediate. No fitting, checkpoint choice or alpha tuning.

## Intervention and identities

Set bridge projection to alpha times its saved weight W, with all other tensors
fixed. Use torch.func.functional_call, never mutate the saved checkpoint.
At alpha=0 compute s=dp/dalpha by autograd. Alpha=1 reproduces the trained
endpoint. Recurrent dynamics are fully replayed, not replaced by a linear head.
Saved float32 alpha=0/1 outputs must reproduce bound_start/fixed_immediate
exactly for every query. Different identities, roles or hashes stop the audit.

For diagnostic precision, replay a float64 copy and compute p0, p1, s there.
Both double endpoints must agree with float32 within 2e-5*(1+abs(reference)).
Report maximum precision changes; do not silently use double as a new model.
First occurrence of each target in each fixed fold/role (chosen without error
or label inspection) additionally checks s against central differences at
h=.001 and .0005. Both must agree within 1e-6*(1+abs(s)). Negative alpha is
only a numerical derivative check, not a deployed or selected intervention.

With r=y-p0, c=p1-p0, define:

- parameter derivative d = -sign(r)*s, or abs(s) if r=0;
- output-chord derivative g = -sign(r)*c, or abs(c) if r=0;
- nonlinear remainder n = g-d;
- crossing curvature k = 2*max(sign(r)*c-abs(r),0), or zero if r=0.

Then MAE(p1)-MAE(p0) = d+n+k, exactly. n can be negative. The parameter
path is not convex: positive d excludes only an infinitesimal beneficial
step, NOT every finite alpha. Report SmoothL1 parameter derivative too.

## Aggregation and falsification

Retain unsupported targets. Collapse repeated folds within each patient,
then patient equal and source-component equal (components are not necessarily
hospitals). Occurrence weighting is secondary. Independent scalar/fsum
aggregation verifies every metric; the existing E35 piecewise geometry helper
verifies output geometry. Numeric classifications use 1e-10*(1+baseMAE+newMAE),
not a scientific threshold or significance test.

For full-endpoint mean harms distinguish positive/negative/ambiguous d. For
positive chord derivatives distinguish positive versus negative d: the latter
is evidence that a useful initial parameter direction becomes unhelpful in
the finite recurrent path. If most d are already positive on held-out roles,
shrinking the parameter step near zero is not the corresponding local repair.
Keep fit improvements, role reversals and counterexamples; do not use truth
to route patients. No new bootstrap gate or promotion is inferred from counts.

## Validation and limits

Tests: analytic nonlinear paths and zero-residual kinks; actual recurrent
autograd/finite differences; zero-bridge equivalence; inference-mode isolation;
unchanged weights; unequal patient repetitions; invalid/nonfinite inputs;
unsupported targets. Bind protocol, code, tests, parents and private results.
Check frozen main tensors and original checkpoint hashes before and after.
Stop on contract/parity/numerical failures, fix implementation without changing
the scientific question. Save completed role outputs atomically for exact
resumption; preserve at least 10 GiB free, no new large event caches.

This is current masked reconstruction on reused development. It cannot establish
future forecasting, treatment causality, self-learning, new external validation
or a complete generalization repair. Runtime, weights and intervals stay fixed.
