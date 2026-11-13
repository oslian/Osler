# Glucose native update-path sensitivity

Fixed seven whole-real checkpoints, same query context, same historical tokens,
same weights and labels. Original predictions reproduce within1e-5. Then allow
recurrent updates only at groups containing an observed glucose measurement;
keep all simultaneous values and final whole-body context unchanged. This
removes non-glucose lab/rate-only updates, not just their numeric values.
No training or replacement of release behavior. Existing filter recurrence used.

Aggregate: `artifacts/glucose_native_cadence_20260911.json`.
Command: `python -m scripts.audit_glucose_native_cadence`.

| Median patient MAE mg/dL | Original | Glucose-update-only | Significant gain/harm |
|---|---:|---:|---|
| Fit | 32.288892 | 33.354323 | 0/7,5/7 |
| Tune | 33.731561 | 34.042923 | 0/7,1/7 |
| New source | 34.840807 | 36.034311 | 0/7,1/7 |
| Hospital | 42.400697 | 42.759651 | 0/7,0/7 |

This does not support fixing current weights by simply skipping non-target
updates. They are used by the trained model, but dependence does not prove
incremental transferable information. Removing updates also removes content,
changes time decay and shifts the training input distribution. Consequently
this is not an isolated causal estimate of cadence, proof of useful memory,
or a train-matched test of target-conditioned updates. The matched own/whole
comparison remains confounded by input content and active update frequency.

Do not deploy glucose-only masking into a shared multi-target state. It would
discard legitimate other-organ information without demonstrating a gain.
Same potassium-conditioned cohort, source roles, inherited pretraining and
clipped-clock limitations remain. Seeds overlap; no new external lockbox,
conformal calibration, retention or promotion. All runtime artifacts unchanged.
