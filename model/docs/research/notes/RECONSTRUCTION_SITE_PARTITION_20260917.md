# Common History-Safe Source Partition

One deterministic common partition is now frozen. It reserves every hospital
component touched by the original hospital-role patients, then assigns remaining
patient-linked components largest-first to the largest remaining60/20/20patient
deficit. Only patient/site provenance determines assignment: no target values,
prediction errors or target eligibility are consulted.

| Role | Patients | Hospitals |
|---|---:|---:|
| Fit | 536 | 11 |
| Tune | 179 | 11 |
| Evaluation | 178 | 11 |
| Reserved | 870 | 21 |

The reserved role includes194original hospital patients plus676patients linked
to their sites. These are not invalid patients; they cannot be reused for source
fitting under the chosen strict history boundary. All patient and site sets are
disjoint across the four roles, including transitive cross-hospital connections.

33of39catalog targets have at least30patients in every source role. The remaining
six retain explicit counts:

| Target | Fit | Tune | Evaluation |
|---|---:|---:|---:|
| Amylase | 23 | 35 | 24 |
| Anion gap | 3 | 143 | 26 |
| Direct bilirubin | 175 | 28 | 67 |
| BNP | 86 | 22 | 16 |
| LDH | 15 | 38 | 4 |
| TSH | 52 | 12 | 15 |

These are not exactly the same33targets as the earlier per-target feasibility
bound. That audit allowed a different pair of hospitals for each target; this
is one common partition. In particular, anion-gap support is severely site
concentrated. Do not move patients between roles after inspecting model errors
or claim absent physiology from this support failure.

The next training comparison can use the33source-supported targets jointly,
while reporting all39catalog statuses. The identical source partition must be
used for numeric, process-only and placebo arms. Fit-only medians and training,
tune-only checkpoint selection, unchanged masking and all-target reporting are
required. No training has yet been run on this new partition.

Three tests verify transitive reservation, row-order determinism, hospital
isolation and rejection of missing/insufficient provenance. Full actual-cache
build verifies input hashes, shard identities and label support without reading
prediction errors. Original reports/weights are unchanged.

This repairs the prospective fine-tuning partition contract. It does NOT certify
the warm-start checkpoint's historical pretraining independence, turn repeatedly
inspected data into a fresh lockbox, or demonstrate improved accuracy.

Machine report: `artifacts/native_reconstruction_site_partition_20260917.json`.
Patient roles stay in Git-ignored durable storage. No runtime or promotion change.
