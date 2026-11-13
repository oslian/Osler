# Potassium amplitude mismatch diagnostic

Protocol: `docs/research/protocols/POTASSIUM_EVENT_GAIN_PROTOCOL_20260911.md`.
Result: `artifacts/potassium_event_gain_20260911.json`.
Command: `python -m scripts.audit_potassium_event_changes --gain`.

All seven original source partitions completed. Frozen event-window trees and
their uncalibrated pooled scores reproduce before evaluation. A patient-equal
LAD scalar is estimated on tuning patients only, separately for own/full/support.
The weighted-median ratio method follows the existing amplitude-calibration
implementation, but removes its upper bound of one. No global intercept or
future-conditioned groups are used. Zero proposed deltas stay zero.

| Arm | Original median MAE | Scaled median MAE | Significant improvement / harm |
|---|---:|---:|---:|
| Own history | 0.373470 | 0.376113 | 0/7 / 0/7 |
| Full history | 0.369971 | 0.370997 | 0/7 / 0/7 |
| Support control | 0.381141 | 0.383179 | 0/7 / 0/7 |

Persistence median MAE is 0.437907. Both scaled own and full beat persistence
7/7, but scaled full beats scaled own only1/7 and support3/7. Neither comparison
establishes improved incremental whole-context generalization.

Full gains range0.899926-1.259218, aboveone6/7. Own gains range0.775593-1.098628.
The median change in marginal MAEs is not the paired estimate; the significant
counts use patient-level paired bootstrap. Lack of significance is not proof
of equivalence. Source evaluation has been repeatedly inspected and is not a
new lockbox; no external data were read for this diagnostic.

## Interpretation

The large-change retrospective attenuation does not establish a globally
mis-scaled model. Permitting amplification selected on separate tuning patients
does not repair new-patient error. This rules against the tested one-parameter
repair, not all conditional amplitude models or the existence of useful signal.
Future-extreme conditioning, heterogeneous conditional changes, finite samples
and incomplete available history remain distinguishable hypotheses. Do not
multiply predictions by the ratio of observed to predicted extreme movement.

No neural or tree weights were retrained. This is not conformal calibration;
release point sources, intervals, registry, and serving behavior are unchanged.
No promotion or causal claim. The aggregate inherits the retrospective audit's
status string and source-protocol binding; this dedicated protocol documents
the --gain branch, which fits only on source tune, not evaluation truth.
