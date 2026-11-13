# Delayed older glucose and incremental error

Frozen seven native whole/own predictions; pre-anchor48h older-arrival strata
from V2 inventory. Original input hashes verified, unchanged query cohort and
labels. Patient-paired bootstrap2,000; no fitting, threshold search or routing.
Command: `python -m scripts.audit_native_out_of_sequence --accuracy`.
Aggregate: `artifacts/native_out_of_sequence_accuracy_20260911.json`.

| Role/group | Patients per split | Whole-own significant gain/harm |
|---|---:|---|
| New source, delayed older glucose | 105-140 | 0/7,2/7 |
| New source, no delayed older glucose | 207-229 | 0/7,2/7 |
| Hospital, delayed older glucose | 75 | 0/7,0/7 |
| Hospital, no delayed older glucose | 133 | 0/7,1/7 |

Hospital delayed group has numerical whole benefit6/7 (none significant),
whereas no-delayed group has numerical harm5/7. Thus the latest hospital
incremental failure is not concentrated in the suspected delayed subgroup.
It also occurs without that pattern in source evaluation.

This is an association audit, not an intervention: severity and observation
support can differ by group, and no conclusion of harmless delayed observations
or a correct update mechanism follows. It weakens the claim that this pattern
alone is the principal cause, and does not justify deleting/reordering events.
Patient groups overlap across windows and seven seeds; counts not additive.

No runtime, neural, interval, promotion or new external validation changes.
The observed fit-to-new-patient failure remains; this specific timing regime
does not provide a demonstrated repair.
