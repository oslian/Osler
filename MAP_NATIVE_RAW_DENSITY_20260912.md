# MAP Native Raw Versus Snapshot Support

## Matched Support Test

Same query cohort, canonical qualified lab/bedside event caches, latest source
versions available by anchor, actual measured time within6h. Same-time conflicting
or quality-invalid measurements excluded. Count distinct measurement times, not
distinct values; true equal-valued repeats remain. Compare with fixed-lag
second-rounded draw proxies. Seed7 descriptive patient-equal query fractions.

| Variable | Fit raw >=2 draws | Fit snapshots >=2 | Hospital raw >=2 | Hospital snapshots >=2 |
|---|---:|---:|---:|---:|
| MAP | 93.27% | 85.41% | 73.17% | 64.75% |
| Heart rate | 95.19% | 88.34% | 72.81% | 65.08% |
| Glucose | 78.64% | 76.58% | 64.35% | 60.17% |
| Creatinine | 6.72% | 6.60% | 8.11% | 7.90% |
| Hemoglobin | 5.48% | 4.49% | 5.36% | 4.43% |
| Lactate | 1.77% | 1.60% | 1.80% | 1.71% |
| Bilirubin | 0.20% | 0.19% | 1.35% | 1.32% |

## Interpretation

Sparse recent lab trajectories are mostly sparse already in these qualified
source caches, not mainly lost by the lag grid. Dense vital histories do lose
some repeated-draw support. Neither observation establishes forecast error
causation. More events may still be redundant or noisy; a matched numerical
experiment is needed before claiming repair. Earlier event-summary trials on
insulin-triggered MAP queries did not establish stable full-own gains, so do
not represent event-history augmentation as wholly untested.

Historical snapshots may preserve earlier legally known versions that later
become conflicted or invalid. Small snapshot>raw counts are reported, not
silently forced to zero or called future leakage. This comparison is not a
complete source-ID retention audit. The source cache itself is selected and
qualified: this does not inventory every downloaded eICU table or patient.

Six focused tests pass including arrival timing, source conflict, patient
isolation, equal-valued remeasurement, and actual-lookback boundaries. No
fitting, runtime, interval or promotion change. Evidence:
`artifacts/map_native_raw_density_20260912.json`; paired counts remain private.
