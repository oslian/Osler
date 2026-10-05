# Frozen potassium change decomposition

Authoritative aggregate: `artifacts/potassium_event_changes_20260911_v2.json`.
Protocol: `POTASSIUM_EVENT_CHANGE_PROTOCOL_20260911.md`.

Seven source patient splits, unchanged frozen event-window and original trees.
No new fitting or external reads. Pooled scores reproduce V1 exactly.
Groups use the fitting patients' patient-weighted 75th percentile absolute
change (approximately 0.6); evaluation truth assigns retrospective membership.
These groups cannot be used as prospective routing inputs or clinical labels.

| Median across splits | Large actual change | Smaller actual change |
|---|---:|---:|
| Persistence MAE | 1.090041 | 0.260765 |
| Own-history MAE | 0.737997 | 0.282909 |
| Full-history MAE | 0.717920 | 0.266571 |
| Support-control MAE | 0.747930 | 0.276494 |
| Full predicted absolute movement | 0.400635 | 0.209985 |
| Actual absolute movement | 1.090041 | 0.260765 |
| Full direction accuracy, nonzero truth changes | 0.906033 | 0.653270 |
| Full delta correlation | 0.784202 | 0.365816 |

Full beats own significantly in 0/7 large-change and 5/7 smaller-change
comparisons; neither has significant full-vs-own harm. Full beats persistence
7/7 in large changes and 0/7 in smaller changes. These are exploratory subgroup
comparisons, not independent confirmatory trials or promotion gates.
Event summaries versus original full trees improve 2/7 large and 1/7 smaller
groups, not a uniform repair. Conditional patient-equal subgroup MAEs do not
add; the JSON separately checks additive contributions under pooled weights.

## Located defect and interpretation

V1 strict floating-point comparison split nominally identical 0.6 changes
across the boundary. V2 treats isclose(rtol=1e-8, atol=1e-8) as equality.
A regression test covers both representations. V1 remains diagnostic history;
no predictions, pooled scores, patient partitions, or thresholds were tuned.

The observed failure is not simply a persistence switch that never activates.
Large-change direction is often correct but movement is attenuated, while
small changes leave little margin over persistence. This is a descriptive
error pattern, not proof that multiplying predictions will help: conditioning
on future extremes itself can expose regression-to-center behavior. Strong
direction in this selected subset does not prove prospective event detection.

Together with earlier raw-tree transfer, this narrows the unresolved problem
to transferable conditional change correction beyond own history, not solely
JEPA compression. It does not identify a unique biological or statistical
root cause. No clinical stability, causal, interval, or runtime promotion claim.
Release weights, registry, calibrators, and serving behavior are unchanged.
