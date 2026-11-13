# E40: actual-step protection is not a complete repair

## What changed

E39 showed that E38 PCGrad protection can fail after Adam momentum/scaling.
E40 keeps that SAME optimizer, initialization, pair order, loss, fit exposure,
frozen backbone and one shared bridge. For different-target pairs only, project
the actual parameter step onto both local non-increase half-spaces. Keep Adam's
moments. Do not use selection labels or pick a posthoc target-specific policy.

This is an intervention, unlike E39 accounting. It is not the older one-step
bounded7 BUN probe, and does not repeat the negative moment-clock recipe.
See [fixed protocol](../../../RECONSTRUCTION_ACTUAL_STEP_PROTOCOL_20261005.md).

## Local mechanism: repaired within its declared scope

All five folds completed, two passes of 2048 fit examples each, 10240 optimizer
steps / 20480 example-steps. These are repeated training observations, not that
many independent patients. All53 inventory retained.

| Local diagnostic | Different-target pairs | Same-target pairs |
|---|---:|---:|
| Example-steps | 19772 | 708 |
| Original Adam proposal adverse dots on this candidate trajectory | 1814 | 112 |
| Applied actual-step adverse dots | 0 | 112 |
| Finite SmoothL1 increases | 489 | 112 |
| Finite decreases | 18012 | 596 |
| Near-zero changes | 1271 | 0 |

Projection changed1709/9886 cross-target updates. Stored float32 rounding
violated the fixed first-order constraint on441 updates; these were explicitly
vetoed, preserving the previous weight. This is a cost of this fixed method,
not hidden skipped data. Same-target pairs were deliberately not protected.

The489 cross-target finite increases are tiny: median1.863e-9, maximum1.192e-7
normalized SmoothL1, none above1e-6. Total positive change2.540e-6 versus total
negative change-2.350485. These descriptive magnitude checks do not modify any
acceptance threshold. Local derivatives still do not guarantee exact finite
loss non-increase. They also say nothing by themselves about unseen patients.

Average cross-target step norm is .003109, versus raw Adam .003251. Hence the
intervention also restricts learning progress/regularization. Do not claim
pure momentum deletion causality, or that zero local harm should imply better
generalization. The raw proposals here belong to the NEW trajectory, not E38's.

## Actual capability: no complete screen

All20480 fixed queries completed:10240fit,5120selection,5120same_source.
Patient-collapsed, target-macro normalized MAE (lower better):

| Role | Actual-step candidate | E38 PCGrad | E38 paired mean | Strong raw |
|---|---:|---:|---:|---:|
| Fit,47 supported targets | .749089 | .748895 | .748914 | not available |
| Selection,46 | .876494 | .876528 | .876553 | .853856 |
| Same-source,44 | .777282 | .777131 | .777108 | not available |

Selection improvement over PCGrad is only about0.004% in this descriptive
aggregate. Fit and same-source means worsen. None of the TEN comparator
screens pass, and their full intersection is empty. No weakened baselines,
threshold adjustment or inference of clinical accuracy from normalized MAE.

### Where this version first fails

- Against PCGrad, only15/47 fit targets have a lower patient mean;32 already
  fail to improve within fitting. This is not solely cross-database shift.
- Selection has26/46 numerical improvements;4 fit gains fail to carry over.
  Same-source has15/44 numerical improvements;8 fit gains fail to carry over.
  Numerical signs are not significance or independent replication counts.
- Against matched initialization,40/47 fit means improve, but18 fit gains do
  not improve selection. Thus fitting does learn, but neither overall fit
  optimization nor conditional transfer is solved by the local constraint.
- The raw comparator remains appreciably stronger on selection. This does
  not by itself prove irreversible encoder information loss (E5 still applies).

## Verification

- Training65018 and independent verification16188 both exited0 naturally.
- Five saved models reloaded; all20480 predictions explicitly replayed with
  maximum difference0. Non-bridge parameters remain bitwise frozen.
- Original E38 identity/truth/control columns unchanged; disjoint adaptation
  roles checked. Existing encoder exposure limitations remain unchanged.
- All saved finite-loss formulas independently reconstructed, fit order and
  support checked, patient/component scalar statistics and all screens rebuilt.
  Maximum aggregation/bootstrap roundoff3.918e-13; science tolerance unchanged.
- 58 experiment and69 verification bindings checked again after completion.
- 28 relevant tests pass, including SciPy constrained-solver comparison,
  degeneracies/rounding veto, disabled-correction exact E38 parity and tampering.
- Fold0/2 weights are identical; fivefold is not five independent replications.

Artifacts: [assessment](../../../artifacts/native_actual_step_20261005.json),
[independent verification](../../../artifacts/native_actual_step_verification_20261005.json).
Private trace files are `actual_step_20261005/fold*/training.parquet` under
the existing ignored research root; no patient rows or IDs are published.

## Decision and remaining problem

Close this fixed branch. The gap between protected gradient and actual update
has been addressed locally; it was NOT a sufficient explanation of failed
whole-body reconstruction. Do not simply remove momentum or sweep protection
thresholds after inspecting these outcomes. The veto/step-size tradeoff is a
limitation, not permission to tune against the same selection set.

Still unresolved: why useful training corrections cannot reliably improve the
full target distribution and transfer to new patients. Conditional support,
representation/readout limitations, objective tradeoffs and finite fit budget
are not fully separated. A future intervention must discriminate these rather
than repeat local non-harm as the success criterion.

This experiment concerns current masked reconstruction, not future forecast,
self-learning deployment or treatment causality. MIMIC/NWICU are reused
development, not untouched lockboxes. No runtime/weights/interval/promotion
changes. The full root-cause and generalization-repair goal is NOT complete.
