# Potassium-native availability anchors

One unresolved contract is that potassium labels were attached to queries
eligible for glucose future scoring. Reconstruct queries at each potassium
result availability, independent of future glucose support. Current potassium
must still have measured age <=6h; future is a measured5-7h result. Preserve
the existing lawful snapshot/history functions and identity/version handling.
Never use measured time as the time a result became known.

Restrict source patients/stays to the prior eligible potassium cohort and
external patients/stays to the prior external potassium comparison. Preserve
the source fit/tune/evaluation patient roles, not repartition by new labels.
Source-only fixed leaf counts and all other settings come from the original
tree search. Fit21 models, three arms x seven seeds, on native anchors only.
No hyperparameter search. Own/full/support share pairs and patient weights.

On the SAME native evaluation rows, compare native-fit and old-fit models,
full-vs-own, full-vs-support and persistence. Source evaluation first, then the
fixed hospital/MIMIC protocols. Assert source patient exclusion and original
model hashes; verify input current equals future_pairs current. Report query
overlap and support; do not compare old-task and new-task MAEs as paired gain.

This changes training anchor distribution/row count, not model architecture.
It cannot isolate each of those changes. Existing patients, MIMIC and source
evaluation have been inspected; not a pristine lockbox. No posthoc filtering,
unit routing, calibration, runtime/promotion, or physiological causality claim.
Patient-equal MAE,1000 patient bootstrap, all seven fixed seeds; unadjusted
descriptive CIs. MIMIC care-unit groups >=30 patients reported separately.
