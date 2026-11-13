# Magnesium-specific availability anchors

The prior Mg task inherited potassium/rate-selected query times. Within the
same available stay universe, rebuilt queries at distinct Mg availability
times and rebuilt actual 5-7h future pairs (current max age 6h). This is not
full-eICU recruitment; raw cache itself was restricted to K-paired stays.

| Support | Prior matched queries | Mg availability queries |
|---|---:|---:|
| Paired rows | 2366 | 2510 |
| Patients | 839 | 809 |
| Distinct future events | 1952 | 2493 |
| Median current age, hours | 3.1667 | 0 |

Only 636 future event IDs overlap. Thus a score change cannot be attributed
to freshness alone; query/label/patient composition changes. New anchors may
still expose older draws because result entry can be delayed. All original
patient roles are preserved and disjoint. Native hospital role:104 patients,
348 rows. This is not a randomized paired timing repair.

Rebuilt truthful whole-lab histories, checked current values against labels,
and ran the same tree leaf grid/source-tune selection/patient-equal metrics.

| Median MAE | Own | Full | Support | Persistence |
|---|---:|---:|---:|---:|
| Source evaluation | .216990 | .217517 | .221144 | .264678 |
| Hospital | .222482 | .215654 | .219527 | .247743 |

Full-own significant gain0/7source,1/7hospital, no significant harm. Full-support
gain0/7source,2/7hospital. Full-persistence7/7both; own-persistence7/7source,
6/7hospital. Repeated splits are overlapping. A small hospital signal appears,
but no stable own-history increment and no generalization repair established.

This tests a raw non-neural comparator, not retrained native/shared JEPA.
No fresh database lockbox, calibration, runtime change or promotion. Private
queries, labels, and predictions stay outside Git. Aggregate reports:
`artifacts/magnesium_anchor_support_20260911.json` and
`artifacts/magnesium_native_anchor_tree_20260911.json`.
