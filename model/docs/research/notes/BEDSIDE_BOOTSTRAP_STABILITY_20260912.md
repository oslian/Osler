# Common-draw patient bootstrap

Added paired_model_errors using existing patient-equal estimate. It requires
common finite predictions and patient IDs, uses identical patient draws for
each contrast, and is independent of display names. Rename/row-order and
missing-prediction tests pass. Existing historical estimators are not rewritten.

Re-evaluated all7bedside partitions with20,000draws and three fixed random
streams20260912/13/14. All three streams agree on gain/harm counts:

| Contrast | Source | Hospital |
|---|---:|---:|
| Real vs lab |1/0|0/1|
| Support vs lab |1/0|1/0|
| Real vs support |1/0|0/1|

The previously salt-sensitive second hospital support-lab pass at1000draws
does not persist. Predictions/MAEs unchanged. This is numerical assessment
stability, not model repair or a new promotion criterion;20,000draws cannot
remove sampling uncertainty or bias from repeated validation use. No patient,
model, threshold or favorable stream was selected. No runtime/promotion.

Authoritative bound report: bedside_common_bootstrap_20260912_v2.json under
artifacts. V1 has identical evaluation but lacked provenance bindings; retained
as intermediate, not authoritative. Dependencies checked against fit-gap report.
