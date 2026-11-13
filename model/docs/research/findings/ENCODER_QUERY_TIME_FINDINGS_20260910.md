# Requested Time Contract Diagnostic

The availability encoder input builder hardcodes6h although glucose labels
accept actual5-7h. Source original-development actual delta quantiles are
5.0/5.533/5.9/6.3/7.0h;65.64% differ from6h by more than15min.

Added explicit optional query_horizons to build_arrays. The argument means
requested prediction time, not a feature describing when clinicians will
measure. It is never automatically inferred from future labels. Tests verify
finite positive shape, and unchanged physiology/history when only time changes.
The default remains6h; active runtime and candidate weights unchanged.

## Fixed-Weight Source Diagnostic

Four requested times (5h,6h,7h,retrospective actual) on14 own/whole identity
2250patient64epoch checkpoints. Original source-tune nominal scores reproduce.
No external inputs or results used. Same136 original tune patients per seed.

Median patient-equal MAE: own36.38397 nominal vs36.37863 actual;
whole36.14475 vs36.13293. Patient-bootstrap improvements0/7each.
Mean absolute5->7h prediction response median0.707own/0.837whole mg/dL.
These are source-selection-data diagnostics, not an independent gate.
Do not compare own/whole medians as paired seed gains.

## Decision

Changing requested time at inference under existing weights does not establish
an accuracy repair. The nominal-time training contract remains imprecise;
training with actual requested elapsed-time supervision is still untested in
this corrected availability-safe experiment. Fixed-weight response cannot
rule out a trained response. A prospective fixed6h query must stay6h; giving
the model a subsequently observed measurement time changes the evaluation
question and cannot be advertised as deployable foresight.

303 tests pass. No promotion, interval calibration or serving change.
Evidence: artifacts/encoder_query_time_20260910.json.
