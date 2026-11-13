# MAP Native Input History Density

## Scope

Descriptive seed7 patient-equal fractions of eligible queries. Count distinct
second-rounded measured-time proxies recovered from observed masks and ages,
not distinct values: equal-valued remeasurements remain distinguishable.
Only measurement times within anchor minus6h through anchor count as recent.
Fixed-lag snapshots may miss intervening raw events, so these are MODEL INPUT
support numbers, not complete raw event rates or independent source IDs.

| Variable | Fit: >=2 recent draw proxies | Held-out hospital |
|---|---:|---:|
| MAP | 85.41% | 64.75% |
| Heart rate | 88.34% | 65.08% |
| Glucose | 76.58% | 60.17% |
| Creatinine | 6.60% | 7.90% |
| Hemoglobin | 4.49% | 4.43% |
| Lactate | 1.60% | 1.71% |
| Bilirubin | 0.19% | 1.32% |

The input contract does NOT contain equally rich dynamic histories for all
organs. Many lab channels supply occasional levels, carried observations and
support patterns, not repeated recent trajectories. This limits claims that
full-body dynamics were exhaustively tested. Dense HR/glucose/MAP histories
also exist, however; sparse lab support alone does not explain every failure.
No outcome-stratified selection, fitting, forecast effect or promotion claim.

## Audit Correction

V1 counted distinct draws visible at recent lag positions, which could include
draws older than6h relative to anchor. V2 enforces actual inferred draw time.
V1 is superseded for recent support, not model training or forecasts. Five
focused tests pass, including carried-draw dedup, equal-valued new draw support,
and exclusion of old draws. Report:
`artifacts/map_native_history_density_v2_20260912.json`.

No runtime, weights or calibration changed. Next distinction is raw-event
availability versus fixed-lag omission; do not claim absent raw observations
from this snapshot audit alone.
