# MAP Native Clock Scope

The current MAP shard builder calls `encode_group`, which retains the historical
clock clipping to [0,168] hours. The earlier potassium clock-retraining result
does not mean this MAP input was repaired. The MAP controls deliberately kept
the old coordinates for matched comparisons.

All seven partitions were audited without fitting. Seed7 patient-equal means:

| Role | Consecutive active clocks with zero encoded delta | Entire history at a clock boundary |
|---|---:|---:|
| Fit | 23.91% | 10.17% |
| Tune | 22.55% | 9.43% |
| New source patients | 24.71% | 11.57% |
| Held-out hospitals | 18.35% | 8.30% |

These are per-query fractions averaged within patient then across patients,
not unique-patient percentages. Among cached nonpadding tokens, 3.03% have
clock0 and 34.50% clock1. Token weighting differs from patient weighting.
Zero differences can also include float32 rounding; the audit does not assert
every zero difference is caused by clipping. Saturated clocks cannot represent
elapsed durations even while the filter continues to perform event updates.

This is a confirmed inherited representation limitation and a repair not yet
transferred to this MAP task. It is NOT evidence that time saturation explains
the accuracy gap: most histories are not entirely saturated, and the prior
potassium clock repair did not produce stable accuracy improvement. Actual
event-time recovery and matched MAP retraining remain required. Do not claim
clock repair alone is sufficient or reclassify previous negative experiments.

Evidence: `artifacts/map_native_clock_scope_20260913.json`. No fitting, runtime,
calibration or promotion changes.
