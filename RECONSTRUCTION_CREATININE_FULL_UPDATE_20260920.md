# Full Source Creatinine Update Trace

Fixed bounded7, all90 selected encoder patients and3095 broad encoder patients
with accepted creatinine examples. No fit patients, duplicated patients or
evaluation patients added. Final outputs reproduce the original source scores
on selected90 and original broad83 within1e-7. Every trace reproduces normal
forward prediction. All variable-presence strata specified by the canonical
schema; groups overlap and are descriptive, not new learned update rules.

| Source scope | N | After-minus-before update MAE | Descriptive paired95% interval |
|---|---:|---:|---:|
| Selected | 90 | -.017695 | [-.122310,.084454] |
| Broad all | 3095 | -.075367 | [-.089266,-.061617] |
| Broad original | 83 | -.116687 | [-.224012,-.010267] |

Negative means lower normalized error.4000patient bootstrap draws; selected
checkpoint and repeatedly examined source population, not formal promotion.
Broad before MAE1.273694 -> after1.198327; final aftergap1.198357. Only25broad
and1selected examples have a positive post-update gap. In broad source, no
last-variable-present stratum with at least30patients has positive mean update
error change. Overlapping strata are not causal organ effects or multiplicity-
adjusted evidence; absence of mean harm does not prove every patient benefits.

## Correction To Small-Probe Hypothesis

The earlier eight-per-group creatinine harm DOES NOT replicate. Even extending
only to all original83broad selection patients reverses it. Final observation
updates are on average useful on this larger source population. Do not disable
the update, add an oracle skip gate, or declare forgetting from the tiny probe.
The selectedgroup interval includes both benefit and harm; not equivalence.

This narrows the remaining problem: the body pathway is connected and its final
update can improve reconstruction, yet incremental model-to-model transfer is
not stable. Those are different questions. All-target generalization, earlier
history retention and remaining six untrained heads are still unresolved; this
does not replace those requirements or establish a complete body model.

Evidence: artifacts/native_reconstruction_creatinine_full_update_20260920.json.
Private paired records stay under ignored data/. No fitting, runtime, interval,
weights or promotion changes.
