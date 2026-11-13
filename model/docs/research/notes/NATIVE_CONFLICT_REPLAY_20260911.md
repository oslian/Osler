# Query-time native history repair

Implemented `conflict_checked_history_schedule`: at each query, retain only
then-known latest versions per lab source and exclude disagreeing same-draw
sources. Resolve source disagreement before applying the48h history window.
Non-lab events retain their original timestamp contract. This explicitly
rebuilds history for a fresh encoder replay, not a mutation of an already
cached recurrent state. Earlier predictions are never revised retroactively.

Synthetic tests verify prefix invariance, conflict onset, resolving revisions,
distinct same-valued remeasurements, physical zero, and earlier conflicting
sources outside the window. An unresolved draw remains absent; a resolving
revision is usable only when it arrives. Lab source identity cannot change
variable or measurement time across versions.

Actual original canonical event table reproduces exactly. Rebuilt5526query
schedules introduce no event outside the original query history.817queries/
344patients change, removing3435event occurrences. This includes obsolete
source versions as well as conflicts, so it is NOT the514query exposure count
and not3435 unique bad measurements or prediction errors.

Private schedule: `native_schedule_conflict_replay.npz` in the existing private
rate-query cache. Old caches retained. Aggregate artifact:
`artifacts/native_conflict_replay_20260911.json`.

Native tokens, matched training and accuracy evaluation remain pending. New
training must use the repaired current/future contract as well, with the same
clean evaluation rows in old/new arms. The schedule repair alone does not
establish forecast gain or repair all generalization failures. No release
weights, registry, intervals or promotion changed.
