# Event-balanced potassium diagnostic

Only intervention versus matched SmoothL1 reference: uniform patient, then
uniform distinct (stay, future source ID), then uniform legal query for that
outcome. Preserve every query and evaluation weight. Rebuild source IDs and
require exact equality of current/future/mask/age/delta against original labels.
Future IDs are training grouping provenance only, never model inputs.

Same seven seeds, source partitions, own/whole capacity, warm start, source-only
support, 64 epochs x16 batches x32, optimizer, JEPA and checkpoint selection as
POTASSIUM_MATCHED_ENCODER_PROTOCOL_20260911.md. Compare saved original predictions
and persistence on identical source-tune patients with paired bootstrap1000.
Primary question is sampling benefit in each arm; whole-vs-own is secondary.
No new external reads or promotion. Tune-selected intervals are descriptive,
not independent confirmation. Pretraining overlap remains unexcluded.
Event balancing changes the training estimand and finite-compute exposure;
failure does not prove repeated labels harmless in every model or target.
