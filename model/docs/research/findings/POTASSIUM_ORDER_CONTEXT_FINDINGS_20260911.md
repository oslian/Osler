# Order issuance does not provide stable incremental repair

Completed28 fixed-capacity source fits on original6174pairs/1309patients,
same seven disjoint-within-seed fit/tune/evaluation roles. Frozen own/full
event-window baselines reproduce. Medication order events replace infusion
snapshots; no external data, model selection, calibration or runtime changes.

Patient-equal MAE medians:

| Input | Original | Real order context | Shuffled order context |
|---|---:|---:|---:|
| Own history | .373470 | .370643 | .373868 |
| Full history | .369971 | .370717 | .370692 |

Real vs original and real vs shuffled: significant improvement0/7 and harm0/7
for both modes. Both real arms beat persistence7/7 (median .437907), which
does not establish order-context benefit. Small numerical change is not a
stable repair; overlapping seeds and exploratory CIs are not independent proof.

## Contract repair verified

Order features ignore final cancellation flag, planned start/stop, route and
dosage. Synthetic cancellation/dose/stop mutation leaves order records unchanged.
No dose inferred; only issuance count/recency enters the existing extra-context
layout. Eventually canceled orders are retained, because cancellation time is
unknown. Thus prior coverage counts excluding canceled orders are not the
training contract. MedicationID dedup; no future order times accepted.

[Official medication documentation](https://eicu.mit.edu/eicutables/medication/)
defines orders, not administration, and describes pharmacy synchronization or
manual entry. Independent entry/availability is absent, so this remains a
retrospective event-time trial, not a proven as-known deployment pipeline.

The result does not establish complete treatment-intent absence: indication,
delivered dose, actual execution and revised-order availability are not tested.
It does reject treating count/recency append as an untested obvious repair.
No runtime promotion. Data and capacity remain fixed, not tuned to make a pass.

Artifact: `artifacts/potassium_order_context_20260911.json`.
Protocol: `POTASSIUM_ORDER_CONTEXT_PROTOCOL_20260911.md`.
Runner: `python -m scripts.audit_potassium_treatment_context --orders`.
