# Matched native conflict-repair training

All28fits complete:7original patient partitions x old/clean inputs x own/full
context. Every arm uses identical combined-clean glucose current/future labels
for fit, tune and evaluation. Thus no comparison across different test truths.
Inputs differ in snapshots and query-asof replay; this is a combined contract
repair, not an isolated event-removal intervention. Source-only support and
event scaling; same32epochs, patient sampler, optimizer, identity value decoder,
EMA target JEPA loss, and source-tune selection including epoch0.

| Median patient-equal glucose MAE | Old own | Old full | Clean own | Clean full | Persistence |
|---|---:|---:|---:|---:|---:|
| Source evaluation |34.284730|34.858198|34.361158|34.884272|56.779125|
| Hospital |42.414126|41.740440|42.656249|42.079848|77.167846|

Patient paired95%CI significant gain/harm counts:

| Contrast | Source gain/harm | Hospital gain/harm |
|---|---:|---:|
| Clean full vs old full |0/1|0/2|
| Clean own vs old own |2/2|0/1|
| Clean full vs clean own |0/2|0/0|
| Old full vs old own |0/2|1/0|
| Clean full vs persistence |7/0|7/0|
| Clean own vs persistence |7/0|7/0|

All counts are out of7 overlapping partitions, not independent replications.
Hospital evaluation has189patients. All54 recorded bindings verified after
completion. Original older scores use different fit eligibility/selection
labels and must not be substituted as matched controls here.

Decision: the known conflict/input repair is not a stable forecast repair.
Keep correct as-of semantics; do not recover old numbers by knowingly accepting
contradictory reports. This rejects sufficiency of this particular repair,
not all measurement-quality explanations or all cross-organ information.
Next causal attribution of model failure must distinguish useful earlier
evidence removal from remaining cross-context overfitting; do not repeat the
same combined repair as untested. No new untouched database, calibrator,
multi-target retention gate or persistent-organ architecture validation.

This is the matched native base-JEPA diagnostic, not full runtime promotion.
No release weights, registry, intervals or serving behavior changed.

Artifacts: `artifacts/native_conflict_replay_tokens_20260911.json`,
`artifacts/native_conflict_repair_training_20260911.json`.
Private checkpoints and row predictions remain outside the repository.
