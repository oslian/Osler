# Native potassium anchor result

Completed21 source-only tree refits with previously frozen capacities. Patient
roles were inherited, not reselected from new labels. Queries now arise from
potassium result availability, not eligibility for glucose future scoring.
Current age <=6h and measured future5-7h remain unchanged. Input current values
match reconstructed labels. Old weights were hash checked and evaluated on
the SAME native rows as new weights.

## Support changed, not just timestamps

| Role | Old patients / pairs | Native patients / pairs | Exact shared queries |
|---|---:|---:|---:|
| Source | 1309 / 6174 | 1087 / 3826 | 2744 |
| Hospital | 207 / 819 | 171 / 459 | 332 |
| MIMIC | 859 / 5674 | 857 / 7154 | 5072 |

Patients/stays were restricted to the old cohort; some no longer have qualifying
pairs. No new patient pool was introduced. The intervention changes sampling,
training row count and current freshness together. It does not isolate time
representation alone. Median current age .55/.50/.6833h, respectively.

## Same-row comparisons

Median patient-equal MAE across seven source seeds:

| Role | Persistence | Native own | Native full | Old own | Old full |
|---|---:|---:|---:|---:|---:|
| Source evaluation | .453487 | .389973 | .374312 | .381978 | .378801 |
| Hospital | .546553 | .460235 | .450165 | .458981 | .446742 |
| MIMIC | .436846 | .379062 | .380615 | .378541 | .379945 |

Native full vs native own gains4/7 source,0/7 hospital,0/7 MIMIC, with no
significant harms. Native full vs support gains2/7 source/hospital but0/7 MIMIC
(MIMIC harm1/7). Both native own and full beat persistence7/7 in all roles.

Crucially, native full versus old full gains0/7 in every role, with source harm
1/7. Native own versus old own gains0/7 everywhere, with source harm2/7. Thus
4/7 full-vs-own source significance cannot be credited to improved full-context
learning; the own control also weakened. The protocol does not support declaring
a repair. External supports and MAEs must not be contrasted with old-task MAEs
as if they were paired improvements.

## Interpretation

Target-native anchoring under this fixed training setup does not restore
stable incremental cross-context transfer. Glucose-conditioned query selection
is not established as the principal cause. The result does not prove all
anchor/cohort choices equivalent or rule out an optimized native learner.
No new hyperparameter selection, care-unit routing, calibration or promotion.

MIMIC and source evaluations are reused; overlapping seeds and unadjusted
bootstrap CIs are descriptive. This is an encoder-free diagnostic, not a
replacement architecture or a verified physical body state. Lab-only context
does not represent all treatment/bedside events. Runtime unchanged.

Protocol: `POTASSIUM_NATIVE_ANCHOR_PROTOCOL_20260911.md`.
Artifact: `artifacts/potassium_native_anchors_20260911.json` (complete=true).
Command: `python -m scripts.audit_potassium_native_anchors`.
