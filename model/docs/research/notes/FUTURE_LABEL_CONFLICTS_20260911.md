# Conflicting simultaneous future labels

Confirmed contract defect in `future_pairs`: distinct source IDs with different
values at the same target measurement timestamp were sorted by ID, then one
was silently used as truth. Native input token construction already rejects
conflicting simultaneous values, so labels were more permissive than that input.

Repair: after resolving revisions per source ID, mark disagreeing same-time
groups. Select the nearest time exactly as before, but reject its truth if
conflicted. Do not skip to a more distant draw. Equal-valued distinct sources
remain valid; known-time current snapshots and final-version semantics unchanged.
Synthetic conflict, shuffled order, farther-draw avoidance, equal-value support,
revision, missing and zero tests pass (4 tests in the label suite).

Real raw cache:2119 conflicting draw groups. On unchanged5526queries,
rejected future pairs:glucose70,bicarbonate78,creatinine12,BUN11,calcium11,
chloride11,potassium11,sodium15; a few other targets1-2 each. No retained future
value, time, ID or current/age changed. No new eligible rows introduced.
Magnesium scored pairs:0 rejected, so this is not the explanation of the Mg gap.

New cache stored separately as `multitarget_labels_conflict_checked.npz` outside
Git. Old caches/checkpoints preserved for historical reproducibility. Source
hashes changed intentionally; older experiment code bindings are historical,
not a reason to bypass their fail-closed checks. No retraining, accuracy gain,
interval recalibration or runtime promotion established. Current snapshot
same-time disagreement handling remains a separate issue, not repaired here.

Artifact: `artifacts/future_label_conflicts_20260911.json`.
