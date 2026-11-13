# Event-summary conflict consistency

The summary path previously counted same-time source disagreements as numeric
excursions, even though snapshots and future labels now reject them. Repaired
the existing helper after as-of revision resolution: all disagreeing sources
at that variable/time are excluded from numerical summaries and valid counts.
Older valid history remains usable; a missing latest snapshot is not replaced
by a claimed current value. Later resolving revisions affect later queries only.
Counts represent accepted events, not all charting activity.

Across5526originalqueries:274 (4.96%) have an as-of conflict inside6h,
648 (11.73%) inside24h. Glucose144/328queries, bicarbonate112/239,
Mg1/1 respectively. These are affected summary windows, not measured forecast
errors or proof conflicts dominate generalization. Source identities and
same-value repeated measurements remain distinct.

25 input/label/replay/summary/preanchor tests pass, including removal of a
false maximum, future revision exclusion, resolution and all-conflict missing
support. No recalculated model scores, native token contract repair, serving
change or promotion. New summary models must rebuild features and retrain;
old features/weights are historical and should not be silently reinterpreted.

Artifact: `artifacts/event_summary_conflicts_20260911.json`.
