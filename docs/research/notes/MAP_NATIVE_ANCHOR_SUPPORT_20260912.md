# MAP Availability-Native Support

## Finding

The inherited query builder selects recorded insulin-rate changes, within an
earlier potassium-qualified stay universe. It is not MAP-native sampling.
Removing the potassium future-pair filter did not remove this query restriction.
This is a scope limitation, not evidence of future values entering the encoder.

Using the SAME available stay universe, build queries at every MAP result
availability time. Retain invalid arrivals in query generation; the existing
quality/conflict policy decides pair eligibility. Reuse current age <=6h,
future 6h +/-1h and original patient roles. All old labels, ages, source IDs
and eligibility reproduce exactly under the same event table.

| Quantity | Insulin-record queries | MAP availability queries |
|---|---:|---:|
| Queries | 55,468 | 266,872 |
| Eligible pairs | 33,477 | 205,723 |
| Patients | 1,505 | 1,702 |
| Distinct future events (stay + source ID) | 32,264 | 146,725 |
| Median current measurement age | 60 min | 10 min |

Only 7,461 query timestamps coincide. Future-event overlap is 22,755. New
queries are therefore not a mere row-count expansion of an identical task.
There are 58,998 more paired rows than distinct future events in the new task;
evaluation must preserve patient clustering and inspect outcome reuse.

## Decision

MAP-native sampling is now available for a matched own-history/full-history
comparison. Current results establish support and task difference, NOT forecast
improvement or a unique explanation for the full-context gap. Freshness may
help persistence too. Never claim old/new MAE difference is paired model benefit.
The population remains restricted to the prior stay universe, not all eICU.
Availability is the recorded nursing entry convention, not independently
verified bedside visibility. No training, calibration, promotion or runtime
change was performed.

Evidence: `artifacts/map_native_anchor_support_20260912.json`; private queries,
labels and seven partition files are hash-bound. Patient roles are disjoint.
Next: rebuild matched inputs, compare own/full on identical native labels;
do not silently reuse old query feature vectors.

## Matched Input Reconstruction Completed

`build_map_native_anchor_inputs.py` rebuilt 266,872 queries and their exact
fixed-lag histories from the qualified raw lab/bedside events, in 60 bounded
stay batches. Every eligible current MAP agrees with the label after inverse
scaling. All six numerical/observation fields at the 7,461 common anchors AND
their histories are bit-exact against the prior full-query cache.

Private shards contain global query indices, so original partitions/labels can
be joined without changing patient identity or row order. No sampled subset or
future-conditioned input generation was used. This is fixed-lag reconstruction,
not a complete native-event neural training run. Own/full matched fitting is
still pending; runtime and promotion status remain unchanged. Evidence:
`artifacts/map_native_anchor_inputs_20260912.json`.
