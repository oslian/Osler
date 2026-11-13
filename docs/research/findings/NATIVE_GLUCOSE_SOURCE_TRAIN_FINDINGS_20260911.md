# Source-aware native glucose history: accuracy result

Completed 28 fits, seven source-defined patient partitions, four equal-width
arms. Protocol: `NATIVE_GLUCOSE_SOURCE_INPUT_20260911.md`.
Aggregate: `artifacts/native_glucose_source_train_20260911.json`.
All 86 bindings verified after completion. Zero-initialized extended projection
reproduced initialized six-channel predictions within 1e-4 before each fit.
Old six-channel training trajectories are NOT claimed identical: use the new
equal-width zero-channel controls for paired interpretation.

The same native query times, accepted measurements, source fit/tune selection,
patient-uniform sampling, 32 epochs, EMA JEPA and value loss were retained.
Source names enter only pre-anchor history events, before the six existing
event channels. They do not change which steps update the native state.
Query-context source indicators are zero; the tested information is history.
Bedside/general glucose names are provenance, not certified specimen/assay IDs.

| Patient-equal MAE, mg/dL | Own zero | Own source | Whole zero | Whole source | Persistence |
|---|---:|---:|---:|---:|---:|
| New source patients | 34.2571 | 34.1774 | 34.4388 | 34.2805 | 56.8777 |
| Held-out hospitals | 41.4195 | 42.1301 | 42.8721 | 42.6263 | 77.3038 |

Entries are medians across overlapping seeds, not independent replications.
Patient-level paired bootstrap gain/harm counts (95% CI, 2,000 draws):

| Contrast | New source | Hospital |
|---|---|---|
| Whole source vs whole zero | 1/7 gain, 1/7 harm | 2/7 gain, 1/7 harm |
| Own source vs own zero | 1/7 gain, 0/7 harm | 0/7 gain, 1/7 harm |
| Whole source vs own source | 0/7 gain, 2/7 harm | 0/7 gain, 0/7 harm |
| Whole zero vs own zero | 0/7 gain, 1/7 harm | 0/7 gain, 1/7 harm |
| Whole source vs persistence | 7/7 gain | 7/7 gain |

## Decision

Retaining source names fixes a representational omission but does not produce
a stable incremental accuracy repair under this protocol. The earlier negative
current-source offset result did not settle this question; this end-to-end
history training now tests a distinct repair and also fails stable benefit.
Do not discard correct source metadata, but do not enable this candidate.

These results do not prove assay differences are irrelevant or that no richer
observation model can work. They rule out claiming this particular source-name
omission was the sufficient explanation for the remaining whole-context gap.
No shuffled-source control or source-dependent measurement-error model was
tested. The original potassium-selected cohort and uncertified historical
pretraining membership remain limitations. No new untouched database was used.

No runtime weights, registry, intervals, or promotion changed. No causal claim.
The model still predicts better than persistence here; the unresolved question
is transferable added value beyond the patient's own history.
