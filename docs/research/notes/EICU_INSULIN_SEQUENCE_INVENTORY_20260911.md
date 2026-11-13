# Full-table sequence support exceeds bounded diagnostic support

Outcome-free full eICU infusion inventory:267937compatible insulin rate events,
same exact original label and distinct nonconflicting times <6h apart.
6144stays/5152patients/67hospitals have sequences;5825stays/4909patients/
66hospitals have a changed rate. No dose integration or availability inference.

Existing diagnostic stays include145such stays/121patients, anywhere during
their stay, NOT necessarily before an eligible potassium anchor. Outside those
stays are5999stays/5037patients. Patient sets may overlap across stays, so counts
are not additive or a ready independent holdout. All counts are inventory only.

This establishes that current rare aligned-subgroup support is not the entire
database's insulin sequence ceiling. It does NOT establish usable paired future
potassium labels, independent known-time support, or an accuracy improvement.
The next necessary data operation is availability-based current/future matching
with explicit source patient/hospital roles. Do not tune on MIMIC or call this
inventory a new external lockbox. Adjacent sequence screening is broader than
the per-query single-label feature eligibility; additional filtering may shrink it.

Artifact: `artifacts/eicu_insulin_sequence_inventory_20260911.json`.
Runner: `scripts.audit_eicu_insulin_sequence_inventory`.
No outcome values analyzed, training, calibration or runtime changes.
