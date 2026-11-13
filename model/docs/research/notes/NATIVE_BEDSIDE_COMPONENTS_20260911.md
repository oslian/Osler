# Fixed bedside-component diagnostic

Seven source-rescaled bedside checkpoints, no fitting. Two interventions at
both query and native history: neutralize five bedside numbers to source centers
while preserving masks/ages/cadence, or remove all five observations including
masks/ages/presence. The latter suppresses bedside-only updates through the
existing inactive-event mask; it is not an isolated cadence-only intervention.
Other inputs and weights are fixed. Original predictions reproduce bit-exact7/7.

An initial replay failed exactness by <=0.000092mg/dL because saved float32
numeric transforms were reconstructed as float64. Reconstructing float32 fixed
exactness; tolerance was not relaxed. This is audit-replay precision, not the
forecast-generalization cause. The failed attempt produced no completed report.

| Intervention vs original | Source gain/harm | Hospital gain/harm | Median paired MAE change source / hospital |
|---|---:|---:|---:|
| Neutral numbers |0/1|1/0|+0.11646 / -0.20948|
| Remove observations |0/5|1/0|+1.37517 / -0.85040|

Counts out of seven overlapping splits. Neither intervention helps reliably
across both roles. Source performance depends more strongly on the combined
presence/age/value path than on actual numbers alone; hospital dependence
differs. This suggests nontransferable context dependence, but does not isolate
workflow shortcuts or prove event frequency is the sole cause. Fixed-weight
ablation is also a distribution shift and is not a validated trained repair.

Do not deploy source-center substitution or delete bedside channels based on
these inspected validation outcomes. A matched trained cadence/support control
would be needed to separate information from update-process effects. No new
weights, calibrator, runtime or promotion. Artifact:
`artifacts/native_bedside_components_20260911.json`.
