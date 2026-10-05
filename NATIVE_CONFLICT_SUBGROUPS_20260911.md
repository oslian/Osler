# Repair exposure versus prediction effects

Frozen28fit outputs, parent bindings reverified; no model fitting or new
selection. A query is changed if its raw native replay indices or query input
vector changed. Groups are changed rows, unchanged rows, and patients with no
changed eligible scored row in the corresponding evaluation role. This last
group is not a claim of lifetime absence of conflicting measurements.

Hospital changed histories have40patients; unchanged rows165patients; patients
without a changed scored input149patients. For clean-full minus old-full MAE:

| Hospital group | Median paired effect | Significant gain/harm out of7 |
|---|---:|---:|
| Changed rows | -0.165877 |0/0|
| Unchanged rows | +0.750590 |0/2|
| No changed scored input in patient | +0.925883 |1/4|

Source changed rows52-65patients: clean-full versus clean-own significant
gain0/harm4, median effect+3.505748. Hospital same contrast0/0; do not infer
a universal harmful subgroup or choose routing retrospectively.

The harm is not confined to edited histories. For unaffected raw inputs,
training, model selection and source-estimated treatment standardization may
still differ. Therefore this is NOT proof that weights alone caused harm or
that the input repair itself should be undone. Isolate fixed weights, fitted
transforms and input replay in a cross-evaluation before attributing the gap.
Same outcome task; seven partitions overlap. No new lockbox, interval,
runtime, promotion or proof of all-target root cause.

Artifact: `artifacts/native_conflict_subgroups_20260911.json`.
