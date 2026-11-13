# MAP Tree Leaf Patient Support

Audited seed7,893fitpatients,200trees/3000leaves per arm. Traversed numeric
splits with exact missing-value routing, verified every fitting leaf row count
against sklearn and summed leaf predictions against the fitted model output.
This inspects the actual trained tree, not an imagined support threshold.

| Arm | Minimum patients/leaf | Median patients/leaf | Leaves with <5patients |
|---|---:|---:|---:|
| Own | 13 | 273 | 0/3000 |
| Full | 4 | 175 | 1/3000 |
| Support | 5 | 211.5 | 0/3000 |

Full's below5 leaf accounts for only0.000003347 mean patient-equal tree
assignment mass. Therefore near-single-patient leaves are not a widespread
explanation on this inspected split. Full leaves are generally narrower in
patient support than own, but support statistics alone do not prove their
predictive relationships stable or unstable. This does not test deeper ensemble
interaction memorization, other seeds, or enforcing a patient-support constraint.

No training/threshold/runtime/calibration/promotion change. Internal sklearn
node use is diagnostic-only, version-sensitive, and guarded by exact count/
prediction checks. Evidence: `artifacts/map_native_leaf_patients_20260912.json`.
