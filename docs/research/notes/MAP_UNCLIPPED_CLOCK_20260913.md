# MAP Raw Clock Recovery

All 60 native MAP shards rebuilt from the original event streams using the
same conflict/availability schedule. Every old token and treatment vector was
reproduced exactly before replacing its two time coordinates with unbounded
raw event hours /24 and /168. Signed pre-origin times are retained.

- 266,872 global queries, exactly once with no omissions or duplicates.
- 625,722 history tokens; 234,666 have changed time coordinates.
- All non-clock input coordinates, event values, history indices and pointers
  match old inputs exactly. No history truncation or invented measurement.
- 182 direct shard-report dependencies hash-verified after completion.

New private `unclipped_clock_shards` files include explicit float64 event_hours
for subsequent audits. Original caches/checkpoints were not overwritten.
Query context and labels remain the original matched task. The change corrects
history time clipping only; it does not claim complete treatment availability
or a new physiological clock architecture.

This is input recovery, not accuracy improvement. Matched MAP own/full/control
training on corrected history clocks remains pending. The previous potassium
clock experiment remains a negative/partial accuracy result and cannot be
used either to promise or rule out MAP benefit. No runtime, intervals or
promotion changed.

Evidence: `artifacts/map_unclipped_clock_{000..059}_20260913.json`.
