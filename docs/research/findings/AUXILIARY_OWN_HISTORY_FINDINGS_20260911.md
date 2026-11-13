# Auxiliary Own-History Results

Completed the predeclared source-only diagnostic on all 14 auxiliary targets
and glucose. Existing 21 checkpoints, original source-fit/tune partition and
measured pairs; 210 new small ridge fits (15 targets x 7 seeds x 2 controls).
No external data, runtime changes, neural retraining or promotion.

Median across seven split patient-equal MAEs:

| Target | Auxiliary JEPA | Own-history ridge | Significant JEPA wins / supported splits |
|---|---:|---:|---:|
| Glucose | 36.2401 | 36.3503 | 0/7 |
| Potassium | 0.3699 | 0.3713 | 1/7 |
| Creatinine | 0.2034 | 0.2392 | 2/7 |
| Bicarbonate | 2.2467 | 2.2196 | 1/7 |
| Chloride | 2.6410 | 2.7757 | 0/7 |
| BUN | 3.4338 | 3.3624 | 0/7 |
| Calcium | 0.3563 | 0.3467 | 0/7 |
| Hematocrit | 2.7355 | 2.4659 | 0/7 |
| Hemoglobin | 0.7867 | 0.7775 | 0/7 |
| Sodium | 2.2230 | 2.1685 | 0/7 |
| Anion gap | 2.2873 | 2.0958 | 0/4 |
| Platelets | 26.4444 | 22.3352 | 0/6 |
| WBC | 2.6083 | 2.7500 | 1/6 |

Magnesium/phosphate have no split with 30 tune patients; not counted as failures.
Significant harms against own-history: glucose1/7, bicarbonate1/7,
hematocrit1/7, sodium1/7, platelets2/6. Other supported targets have no
significant harms. CIs are descriptive selection-data patient bootstrap, not
multiple-testing-adjusted promotion evidence.

## Interpretation

Potassium's6/7 persistence wins shrink to1/7 against own history. The evidence
therefore does not isolate cross-organ gain: much of the observed useful
prediction is competitive with a target-only low-complexity reference. Glucose
also has0/7 significant incremental wins. No tested target has7/7 wins.

This does NOT prove that whole-body information is absent or that a specific
encoder component loses it. Ridge and JEPA differ in capacity and training.
The known earliest failure remains fit-to-new-patient generalization; the new
comparison narrows the claimed added capability, not a unique mechanistic cause.
Any next encoder intervention must beat matched own-target learning, not merely
persistence or an unused output head. Do not tune a deployment selector on these
retrospective findings. The current protocol uses glucose-anchored queries, not
all clinical presentations, and checkpoint pretraining overlap is not excluded.

Evidence: `artifacts/encoder_auxiliary_own_history_20260911.json`.
Command: `python -m scripts.audit_auxiliary_learning --own-history`.
Tests cover own-channel isolation and current-only extraction. Existing
availability-contract tests cover history construction; no new temporal schema.
