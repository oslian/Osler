# Query-Conditioned Histories Are Not A Canonical Event Ledger

Attempted to inventory multi-target reconstruction examples by unioning native
tokens per patient, then per encounter. Both strict time checks failed. Exact
same-time deduplication also failed: values/masks/ages/presence differ between
tokens at the same encounter time. No completed support report or training
was produced. The script remains fail-closed at conflicting copies.

Source inspection explains the contract: conflict_checked_history_schedule
selects each source's last revision known at the query, resolves measurement
conflicts, then applies the48h window. The token lookup key is the resulting
source-row group, not just event time. Different query histories may therefore
contain different versions of an earlier group. This is intentional as-of
query replay, not proof of original history leakage or duplicate ingestion.

Unioning these histories to create new earlier target anchors would be invalid:
it can apply later conflict/revision knowledge to earlier examples. Do not
merge conflicting tokens, pick an arbitrary copy, or only compare timestamps.
Patient-heldout membership also does not authorize concatenating encounters.

Required multi-target builder: read original measurement source/version events;
derive new availability anchors; call the conflict-checked scheduler with those
anchors; obtain as-of current target truth and target/sibling masks there.
Keep each encounter separate and fit/tune partitions patient-independent.
This is an unfinished training-data construction task, not an efficacy result.

Sources: scripts/build_map_native_query_tokens.py,
scripts/build_map_unclipped_clock.py,
whole_body_jepa/native_history_schedule.py.
Attempt: scripts/audit_native_multitarget_support.py.
No weights, registry, runtime or calibrators changed. No promotion.
