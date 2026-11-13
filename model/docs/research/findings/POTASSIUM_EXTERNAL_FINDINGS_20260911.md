# Frozen Potassium Transfer Results

All14 fixed checkpoints evaluated, no retraining or selection. Current/future
potassium rebuilt under the same availability/5-7h contract at existing glucose
queries. Source development patient overlap checked and absent. Pretraining
overlap is not excluded; MIMIC is repeatedly inspected validation, not lockbox.

| Role | Patients / pairs | Own median MAE | Whole median MAE | Persistence |
|---|---|---:|---:|---:|
| eICU hospital | 207 / 819 | .43031 | .42972 | .52131 |
| MIMIC | 859 / 5674 | .37945 | .38250 | .42890 |

Whole beats persistence7/7 in each pooled role; whole beats own0/7 in both.
MIMIC whole versus own is significantly worse2/7, hospital harm0/7.
Thus source-tune's consistently signed small whole advantage did not establish
incremental transfer. This is not evidence that potassium cannot be forecast.

## MIMIC Descriptive Care-Unit Comparisons

| Unit | Patients | Whole beats own | Whole beats persistence |
|---|---:|---:|---:|
| CVICU | 173 | 5/7 | 7/7 |
| CCU | 111 | 0/7 | 7/7 |
| MICU | 217 | 0/7 | 0/7 |
| MICU/SICU | 141 | 0/7 | 2/7 |
| SICU | 114 | 0/7 | 6/7 |
| TSICU | 78 | 0/7 | 5/7 |

Groups are first-care-unit descriptive strata, not additional independently
trained heldout experiments. Repeated seeds/patients are not independent trials.
No posthoc CVICU router is permitted from these observations.

## Decision

The same-capacity comparison resolves the prior ridge-capacity ambiguity for
this potassium trial. Whole-context incremental generalization still fails,
with subgroup heterogeneity rather than a universal no-signal result. Root
causes such as treatment distribution, sampling and representation remain
alternatives, not identified causes from these scores. No conformal, retention
or untouched final gate was run. No runtime, weights, registry or interval
change; no promotion. This is one target at glucose-query anchors, not a
complete potassium or whole-body capability audit.

Artifact: `artifacts/potassium_matched_external_20260911.json`.
Protocol: `POTASSIUM_EXTERNAL_PROTOCOL_20260911.md`.
Command: `python -m scripts.evaluate_potassium_matched_encoder`.
