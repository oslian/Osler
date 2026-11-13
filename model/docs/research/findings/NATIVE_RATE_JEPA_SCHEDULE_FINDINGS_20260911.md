# Native history schedule support

V2 materialization completed: native_events.parquet and native_schedule.npz
are now private, hash-bound training prerequisites rather than transient
counts. Unit and measured_at provenance retained; rate measured_at remains
unknown instead of claiming its record time was administration time.
Every serialized query/group/event index round-trips exactly. Counts unchanged.
Artifact `artifacts/native_rate_jepa_schedule_20260911_v2.json`. No neural
training yet. A subsequent performance-only edit loads NPZ arrays once rather
than decompressing them in every loop iteration; original verified report
and caches preserved, equality logic unchanged.

Built and tested a source-namespaced, identity-deduplicated native event schedule.
Same5,526paired queries,48h bounded history. Query timestamps do not add updates;
simultaneous events are grouped, distinct equal-valued events retained, future
events and other stays excluded. Four schedule/event-input tests pass.

Observed native simultaneous-event group counts: minimum3,median42,p9090,p99120,
maximum165. Individual event counts per query: median102,p90172,maximum353.
5,296queries have more than10native update times; no query empty. Canonical
event table779,721rows includes relevant stays outside individual query windows.

This quantifies how the native sequence differs from ten fixed snapshots. It
does not mean all extra events contain independent information, or that a
snapshot loses every event's effect: later snapshots can retain earlier values.
Forecast improvement has not yet been tested. Treatment timestamps remain
record times, whereas lab timestamps are result availability; do not collapse
those provenance semantics into a verified clinical online contract.

Aggregate `artifacts/native_rate_jepa_schedule_20260911.json`. No trained model,
runtime, intervals or promotion. Next requires native-sequence inputs to the
same filter with explicit event updates, not simply more duplicated snapshots.
