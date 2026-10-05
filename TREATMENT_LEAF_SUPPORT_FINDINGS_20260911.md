# A rare-feature comparator cannot test all treatment signals

## All-action support follow-up

V2 extends the same source-only preflight to all six action groups. These
checks are necessary support conditions, not statistical power or efficacy.

| Action | Fit permits pure exposure split /7 | Evaluation has exposure /7 | Fit exposed patients range |
|---|---:|---:|---:|
| Bicarbonate | 0 | 6 | 1-7 |
| Diuretics | 7 | 7 | 10-14 |
| Insulin | 7 | 7 | 110-129 |
| Magnesium | 0 | 2 | 1-2 |
| Potassium | 0 | 5 | 2-5 |
| RRT | 0 | 0 | 0 |

The runner now records action-level row AND patient counts and necessary split
support on future fits. Empty evaluation is not a failed action efficacy test.
This does not change pooled scoring or promotion rules. Old reports remain
immutable; V2 is `artifacts/treatment_leaf_support_20260911_v2.json`.
Insulin is not explained by this rare-leaf obstruction; its previous negative
incremental result still needs separate interpretation. Even diuretic row
support comes from few patients and does not establish sufficient power.

The previous infusion experiment fixed min_samples_leaf=30 from the lab-only
tree. Exact source query support for potassium infusion is20rows in total.
Seven fit counts are11/13/19/13/19/8/9; evaluation9/7/1/1/0/0/4.
No fit can isolate an exposed-only leaf. Two evaluation splits have no exposed
query at all. Counts are windows, not independent patients; effective support
is not larger than these counts.

Synthetic positive control with200rows,20exposed, y exactly equal to exposure:
min leaf30 gives zero predicted contrast and MAE .1. Changing only min leaf
to10 recovers contrast .9999999993 and MAE7.06e-11. This verifies structural
test insensitivity, not a real clinical effect or a forecast repair.

## Correct interpretation

The broad infusion append still did not improve pooled accuracy. But for the
rare potassium channel, lack of tree splits is partly forced by support and
capacity constraints: it is not a successful falsification of its information.
The order trial has different support and is not explained by this result.
This is a diagnosis of this comparator, not a universal JEPA root cause.

Do not lower the deployment gate or call synthetic recovery promotion. A
lower-leaf trial would increase variance, and the exposed external evidence
would remain insufficient. Before a rare-action-specific conclusion, require
fit support compatible with the learner and nonzero independent evaluation
support. Preserve the whole-cohort result separately from action-specific
testability. New data or a larger predeclared source cohort may provide support;
resampling20windows cannot create independent patients.

Artifact: `artifacts/treatment_leaf_support_20260911.json`.
Runner: `scripts.audit_treatment_leaf_support`; regression test includes the
perfect-signal positive control. No real model retraining, runtime or promotion.
