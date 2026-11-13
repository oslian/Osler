# As-of snapshot conflicts

Confirmed and repaired current snapshot selection: the old state used one event
per variable, with availability/source-ID tiebreaks. Distinct sources disagreeing
at the newest measured timestamp could silently become a single observed value.

Snapshots now retain each source's latest known revision, inspect all sources
at the newest measurement time, and emit a value only if they agree. A conflict
remains missing, with its true age retained; no older-value substitution. A later
known resolving revision or newer draw restores an observation. Future revisions
cannot resolve a past snapshot. Equal-valued sources and physical zero survive.

On the old5526query/history grid, retained observed values are bit-identical,
history indices identical, and no previously missing cells become observed.
Affected histories: bicarbonate210/104patients,glucose144/82patients,
creatinine38/13patients; Mg1history/1patient (whole query universe, not necessarily
the Mg scoring subset). These are affected input histories, not wrong forecasts.
New private cache `body_arrays_conflict_checked.npz`; original preserved.

22 availability/label/raw-replay/preanchor tests pass. No training or evidence
of improved forecasting yet. Native event-token history and event-summary
semantics require separate consistency review; this change is the snapshot
builder, not a claim every input path is repaired. Old cached-label current
fields must not be paired with the new inputs without rebuilding eligibility.
No release weights, serving registry, intervals or promotion changed.

Artifact: `artifacts/snapshot_conflicts_20260911.json`.
