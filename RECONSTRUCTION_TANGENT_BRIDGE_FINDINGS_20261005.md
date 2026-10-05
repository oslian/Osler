# E43: tangent fitting is possible, but the fixed repair does not transfer

Current target-masked reconstruction only. This is not future forecasting,
self-learning, a clinical test, or release promotion. Five existing folds were
completed and independently replayed. Fold0/2 saved checkpoints are identical;
these are not five independent replications.

## What Changed

E42 tested the fixed output matrix. E43 instead differentiates the actual
recurrent prediction with respect to the single shared 64x448 immediate bridge
at zero. All other weights remain fixed. Each fold uses the same 2048 fitting
queries and existing 1024 selection plus 1024 same-source queries.

The full fitting Jacobian has 2048 retained numerical directions in every fold,
under the prespecified threshold. The regularized SmoothL1 tangent objective
converges in 24-36 iterations after the disclosed coordinate conditioning.
The original-coordinate gradient infinity norm is at most 1.740e-8.
This excludes a hard local shortage of fitting-output directions in this sample.
It does not establish useful physiological information or predictive learning:
28,672 parameters for 2048 fitting queries can fit sample-specific residuals.

All five full steps fail the fixed actual-model Armijo test; half steps are
the first accepted steps. No selection results were used to pick the step.
The linearized full-step fitting SmoothL1 is 0.012667-0.054807, but that
surrogate cannot be substituted for the actual nonlinear recurrent prediction.

## Actual Truth Error

Normalized MAE, patients equalized within target, then targets equalized.
Lower is better. These are not percentage accuracy or physical-unit errors.

| Role | Scored targets | Zero bridge | E40 protected step | E43 actual model | E43 linear diagnostic at accepted step | Raw control |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 47 | 0.753682 | 0.749089 | 0.745849 | 0.407425 | Not scored in this role |
| Selection | 46 | 0.876270 | 0.876494 | 0.893038 | 1.332393 | 0.853856 |
| Same source | 44 | 0.781413 | 0.777282 | 0.784977 | 1.163262 | Not scored in this role |

Compared with zero bridge, the actual candidate's macro MAE changes by
-1.039% on fit, +1.914% on selection and +0.456% on same-source patients.
These aggregate changes are descriptive, not new significance tests.

Component-equalized actual-model MAE is 0.785489 / 0.940151 / 0.777273,
versus zero bridge 0.795575 / 0.921333 / 0.770998. SmoothL1 has the same
aggregate improvement-on-fit / deterioration-on-query pattern under both
weightings. The result is not explained solely by switching from the training
loss to MAE.

Of 47 scored fitting targets, 34 improve in patient-equalized MAE versus zero
bridge. Among the 33 fitting improvements also scored in selection, 23 lose
their mean advantage. Among the 33 also scored in same-source, 23 lose it.
These counts are not counts of statistically established harms. Target support
and repeated-fold patient aggregation remain unchanged.

All eleven individual comparison screens and their full intersection are empty.
The selection inventory remains 53 targets: 7 unscored, 19 below minimum
support, 27 meeting minimum support. No gate or difficult target was removed.

## Where This Version Fails

1. **Tangent-to-finite-step mismatch on fitting data.** A large surrogate gain
   becomes a small actual-model gain. Finite differences verify the derivative,
   so a valid derivative is not a guarantee of a faithful large-step forecast.
   This does not yet isolate tanh saturation, recurrent feedback, or another
   nonlinear operation. E29's earlier nonsaturation result belongs to a
   different, smaller learned correction and cannot settle E43.
2. **Transfer failure is also present without that mismatch.** Even the
   linearized diagnostic is substantially worse on both query roles. Perfectly
   reproducing that surrogate would therefore not repair this fixed method's
   held-out mean error. Better tangent fitting alone is insufficient.
3. **Actual fit gains also fail on same-source patients.** Changing source
   identity is not necessary for the observed mean reversal. This is not proof
   of a unique cause: conditional support, composition, finite sample and
   representation constraints remain confounded. Original encoder exposure
   is inherited from the existing roles; these are not new untouched patients
   for every upstream component.

The new evidence distinguishes local fitting feasibility from useful transfer.
It does not prove that the state contains everything, that all information is
lost, that all optimizers fail, or that bounded physiology updates should be
removed. Optimization budget and regularization differ from earlier Adam runs;
this is not a pure optimizer causal comparison. The raw reader also retains
its larger fitting support, as already diagnosed in E24/E25.

## Verification And Decision

- Original run34265 stopped at the required 2000-iteration success check before
  any held-out evaluation. Its source snapshot and original protocol remain
  hash-bound. The separate numerical amendment is not hidden.
- Formal run54282 completed all five folds and 20,480 queries, exit0.
- Independent verifier20779 reloaded every checkpoint, checked frozen tensors,
  original roles/controls, stored direction and accepted step, and replayed
  every query: maximum prediction difference 0.
- Scalar patient/component errors, paired statistics and screens agree within
  6.661338147750939e-15; the scientific thresholds are unchanged.
- Directional finite-difference checks cover the first example of each
  represented target/role; maximum error 3.800232604295184e-9. This is not a
  finite-difference test of every parameter and query.
- All 45 training-report bindings and 33 independent-verification bindings
  rechecked. 23 related unittest tests pass; tests are not an efficacy gate.

**Reject this fixed one-linearization repair for deployment.** Do not sweep its
regularization, step sizes or seeds on the same selection results. Any follow-up
must distinguish a mechanism not tested here; merely improving linear fit is
not enough. Full generalization repair remains unresolved.

Runtime, release weights, uncertainty intervals and safety anchors are unchanged.
Research checkpoints remain private; no commit or upload was performed.

## Evidence

- [Fixed protocol](RECONSTRUCTION_TANGENT_BRIDGE_PROTOCOL_20261005.md)
- [Numerical amendment](RECONSTRUCTION_TANGENT_NUMERICS_20261005.md)
- [Complete results](artifacts/native_tangent_bridge_20261005.json)
- [Independent verification](artifacts/native_tangent_bridge_verification_20261005.json)
