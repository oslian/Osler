# MAP Native Query History Integration

The missing new-query-to-neural bridge now has a bounded shard builder:
`scripts/build_map_native_query_tokens.py --shard N`. It uses the existing
conflict-checked, past-only history scheduler and existing event tokenizer.
Source observations, query partitions, training policy and release are unchanged.
Each output stores its exact global query row mapping; no truncation or
subsampling of histories is performed. Existing outputs are not overwritten.

Shard 0 completed: 1,885 queries, 4,464 unique event groups, median history 81
groups and maximum 176. All 19 common old query histories reproduce both token
values and treatment values exactly. Artifact bindings are recorded in
`artifacts/map_native_query_tokens_000_20260912.json`.

The existing `native_arrays` assembler was then exercised on every query in
this shard. Global query mappings match the snapshot shard. All 1,885 assembled
native sequences reproduce the stored ragged sequences exactly; query context
checks pass and all input values are finite. Shapes are 6,350 by 172 for inputs
and 6,350 by 176 for padded histories. This verifies assembly, not a neural
forward/gradient run or predictive accuracy.

Four existing native-history contract tests pass (identity, time, cross-source
conflicts, prefix replay, equal-valued remeasurement and zeros). Diff checks pass.

## Complete Build and Verification

All 60 shards are now built. `scripts/verify_map_native_query_shards.py`
verified every shard's bound source and output hashes and ran the existing
neural input assembler on all shards. Coverage is exactly once for every one
of 266,872 queries, including 205,723 eligible MAP future pairs. All 7,461
common old query histories are exact. Every assembled native token sequence
and treatment sequence is exact, query contexts are unchanged and all numerical
input values are finite. No event-history truncation was introduced.

The machine-readable complete report is
`artifacts/map_native_query_history_complete_20260912.json`.
This supersedes the initial shard-only progress above. It does not certify
pretraining independence, online treatment availability or neural accuracy.
Matched own/full JEPA training on the new MAP task remains unfinished.
The original insulin events retain
record-time semantics, not independently verified administration availability.
Original lab cache qualification is inherited, not a new raw-data completeness
claim. No calibration, runtime update or promotion is implied.
