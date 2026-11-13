# Student training mode does not resolve the context gap

Completed 14 matched fits using the existing Transformer dropout=0.1 during
student training. EMA teacher remains eval; all evaluation deterministic.
Protocol: `NATIVE_GLUCOSE_MODE_PROTOCOL_20260911.md`.
Aggregate: `artifacts/native_glucose_mode_20260911.json`, 68 bindings verified.
All old own/whole predictions reproduced before each fit. Mode assertions ran
every epoch. Old experiments intentionally used deterministic eval-mode
training; gradients were active. This is not a claim of a disconnected graph.

| Median patient-equal MAE, mg/dL | Own old | Own train | Whole old | Whole train |
|---|---:|---:|---:|---:|
| New source | 33.9762 | 34.3637 | 34.8408 | 34.3263 |
| Hospital | 41.4025 | 41.8343 | 42.4007 | 42.1865 |

Paired patient-bootstrap significant gain/harm, out of seven overlapping seeds:

| Contrast | New source | Hospital |
|---|---|---|
| Own train vs old | 1 gain, 0 harm | 0 gain, 0 harm |
| Whole train vs old | 1 gain, 0 harm | 1 gain, 1 harm |
| Whole train vs own train | 0 gain, 3 harm | 0 gain, 0 harm |
| Whole train vs persistence | 7 gain, 0 harm | 7 gain, 0 harm |

Medians across seeds are not paired effects: the slightly smaller pooled
whole-train median than own-train on new source does not contradict the paired
harm counts. Different seeds contribute to the medians. These are exploratory
reused patients, not independent confirmatory replications.

The omitted activation/attention dropout is therefore not a sufficient repair
under this fixed schedule. Prior variable-block dropout was a different
experiment; both now have negative evidence for stable incremental gain.
This does not exclude other regularization, nor prove irreversible lack of
cross-organ information. No probability sweep or posthoc schedule was used.

No runtime weights, calibration, registry, causal or promotion changes.
Historical pretraining membership remains uncertified; potassium-conditioned
queries and record-time treatment caveats remain. No new lockbox was opened.
