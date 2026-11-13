# MAP Native Missing Routing

## Controlled Test

All 14 own/full HGB fits completed. Cloned original estimator settings and
preserved source-only allowed variables, patient-equal fit weights, labels,
partitions, and explicit mask/age features. Only mask-zero numeric entries
become NaN, using the installed sklearn tree's native missing routing.
Observed normalized zero stays zero. Existing mask means the old representation
was not an exact observed/missing collision; this is an inductive-bias test,
not a proven information-loss bug. All original held-out MAEs reproduced.

| Seven-split median MAP MAE | Source evaluation | Held-out hospitals |
|---|---:|---:|
| Original own | 8.78353 | 10.24211 |
| Original full | 8.74362 | 10.37322 |
| Native-missing own | 8.71540 | 10.21838 |
| Native-missing full | 8.71981 | 10.31455 |

Patient-paired 20k bootstrap: full-originalfull gain0/7source,2/7hospital,
harm0both. Own-originalown gain3/7source,0hospital, harm0both. Full-matchedown
gain0/7both, harm2/7source,0hospital. Full-persistence gain7/7both.

## Decision

Partial numerical improvement, not a stable incremental full-history repair.
No source-tune parameter search or test-specific missing rule. Preserve original
release inputs; changing representation requires retrained weights. This does
not establish that other observations carry no information or that observation
workflow is the sole failure cause. A native-missing support arm was not trained,
so numeric-versus-support attribution under this representation is unmeasured.

This is a fixed-lag raw-tree diagnostic, not anatomical JEPA training, interval
calibration, a new lockbox, or promotion. No runtime change. Private checkpoints
include the missing representation metadata. Focused zero/missing/isolation test
passes; source reports and predictions are verified through artifact bindings.
Evidence: `artifacts/map_native_missing_training_20260912.json`.
