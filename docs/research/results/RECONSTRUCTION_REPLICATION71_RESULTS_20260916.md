# Source Recovery And Seed71 Replication

Both missing upstream event files were rebuilt with the original extractors
and the surviving cohort. Their SHA256 values match historical bindings exactly.
No missing-file bypass or altered data contract was used. Evidence:
`artifacts/reconstruction_sources_rebuilt_20260916.json`.
The rebuilt files remain in durable Git-ignored staging and were restored to
the canonical paths only after equality checks.

Seed71 then completed all three eight-epoch arms and heldout evaluation:

- `artifacts/native_reconstruction_replication_v2_71_epoch8_20260914.json`
- `artifacts/native_reconstruction_replication_v2_71_evaluation_20260914.json`

All report bindings verify. Each epoch used25,579 fit patient-target pairs and
35,958 updates per arm (287,664 total). All39 targets meet source support.
Evaluated14,435 pairs under seed71 patient roles and fit-only medians.

| Arm | Selected epoch | Source tune normalized MAE |
|---|---:|---:|
| Real | 7 | 1.870456 |
| Trained process-only | 6 | 1.979857 |
| Placebo | 0 | 2.023692 |

Nominal95% paired intervals favor real over process on19evaluation and8hospital
targets. Seven overlap; six also beat population and placebo in both roles:
albumin, alkaline phosphatase, ALT, AST, heart rate, total protein. Calcium
passes the process comparisons but not all references in both roles.
Selected placebo0 equals population, not an independent confirmation.

Across seeds7/19/31/43/59/71, the combined-comparison intersection remains
albumin, AST, total protein. Hospital lipase significantly harms process-only;
no nominal population/placebo harms appear. Lack of significance is not equivalence.
Hospital fibrinogen/LDH/TSH support remains26/27/17 and unsupported.

Only seed83 remains under the fixed protocol. The hospital cohort is shared
across seeds; tests remain nominal and do not establish final multiplicity-
adjusted or independent external validation. Current reconstruction is not future
forecast or uncertainty validation. No promotion/runtime changes. New seed71
checkpoints and private predictions were also copied to the durable ignored backup.
