# Matched raw-history source diagnostic

Completed105 Ridge fits: seven splits x three inputs x five tune-only alphas.
Same three-way evaluation patients/labels as neural trials. No new external
data. Float32 first pass emitted ill-conditioned solver warnings; identical
float64 rerun removed these warnings and is authoritative. Original aggregate
retained as float32 diagnostic, not an alternate selected result.

| Input | Median evaluation MAE |
|---|---:|
| Own raw history | .376129 |
| Full raw history | .392061 |
| Own numbers plus cross support | .385800 |
| Prior full JEPA | .374244 |
| Persistence | .437907 |

Full raw vs own gain0/7,harm3/7; vs support gain0/7,harm2/7;
vs JEPA gain0/7,harm2/7. Full persistence5/7,own persistence7/7.
Cross-context alphas generally select1000-10000 (full upper boundary4/7).
Thus this finite linear regularization grid is not proof of optimum or
absence of nonlinear information. No posthoc enlargement based on evaluation.

The hypothesis that bypassing the encoder immediately recovers useful raw
cross-numeric information is unsupported by this readout on these same patients.
It is now less justified to attribute all failures to latent compression alone.
High-dimensional fit/regularization and weak conditional information remain
alternatives; no unique root cause or practical repair established. Unequal
feature dimensions and non-equivalent model capacities must be retained in
interpretation. No calibration, runtime weights/registry or promotion changes.

Artifact: `artifacts/potassium_raw_history_20260911.json` (complete=true).
Protocol: `POTASSIUM_RAW_HISTORY_PROTOCOL_20260911.md`.
Command: `python -m scripts.audit_potassium_raw_history`.
