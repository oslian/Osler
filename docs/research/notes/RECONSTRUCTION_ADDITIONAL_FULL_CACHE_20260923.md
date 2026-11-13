# Full Additional-Domain History Cache

## Completed, Not Training

The original process completed all105hospital shards in74,660.1seconds
(20.7hours), without interruption/restart. It started under the20260922
experiment ID and finished20260923. The cache contains1,800,214examples for
73,670people and occupies5,283,131,392bytes (4.92GiB). It is private and ignored.
SQLite integrity check is `ok`.

Existing availability-time replay, source conflict handling, fresh anchor
labels, complete target/sibling masking and48h historical window were reused.
No history was additionally truncated for storage. Values retain raw units;
fit-specific scaling occurs at read time, not in five duplicated caches.

Every generated label matched its accepted raw-label center within float32
conversion tolerance. Every history matched its earlier accepted event-group
count and contained no target/sibling observations. Across ALL examples and
five fold scalers, unsupported observed-input counts are zero. This resolves
the earlier pilot-only support uncertainty for the present vocabulary, not
generalization to unseen databases or future extracted channels.

## Independent Verification

After completion, independently re-read every SQLite patient/target/label row.
The full identity matrix exactly equals the finite entries in the original
patient-center matrix: no substituted, missing or duplicate patient-targets.
All1,800,214labels match within float32 tolerance. Full SQLite SHA256 and all
bound code/input files match. This independently verifies metadata identity and
labels; it is not a second full raw-event re-encoding or payload-by-payload
roundtrip. The prior847example pilot established exact payload roundtrips.

## What Remains

No encoder/readout/calibrator was trained by this cache build. No accuracy,
future forecasting, causal or promotion claim follows from data acceptance.
The next experiment is matched freshly trained physiology versus process-only
encoders with the already fixed component-disjoint fit/selection/calibration/
test roles. Preserve patient-equal sampling and fold-local normalization.
Use the wider domains to test residual transport, not to silently replace the
old selected-insulin cohort whose support remains insufficient. MIMIC/NWICU
remain reused validation, not new external lockboxes.

Runtime weights, registry, intervals and promotion status are unchanged.
Report: `artifacts/native_reconstruction_additional_full_cache_20260922.json`.
