# MAP Explicit Gap Training

Completed 21 fixed HGB fits: remove reported gap; replace its candidate input
coordinate with explicitly named same-time derived no-K gap; replace with
derived support/age only. Original observations are unchanged. The candidate
metadata identifies replacement semantics and physical mmol/L units. Other
coordinates, labels, patient weights, source support, estimator parameters and
partitions are unchanged. No test-selected coefficient or convention assignment.
Derived histories use exact lag timestamps and six-hour maximum draw age.
Original reference MAEs reproduce. Three focused tests pass.

| Seven-split median patient MAP MAE | Source | Hospital |
|---|---:|---:|
| Own history | 8.71540 | 10.21838 |
| Original full (native missing) | 8.71981 | 10.31455 |
| Full without gap | 8.71920 | 10.24189 |
| Full explicit derived gap | 8.71485 | 10.25123 |
| Full derived support only | 8.70711 | 10.26380 |

Derived versus original full or own: significant gains0/7 both roles, harms0.
Derived versus no-gap or derived-support: gain1/7 source,0/7 hospital, harms0.
Derived-persistence gains7/7 both. No-gap versus original full gains0/7 both,
despite all seven hospital point estimates improving. Patient-paired20k95% CI.

The convention concern has a concrete matched test now, not just speculation.
Numerical hospital improvement does not establish the cause or an effective
repair: removing gap performs similarly, and the derived numbers do not show
stable incremental value. The task, observed-support change and deriving
features from existing components limit mechanistic interpretation. Same time
is not specimen identity. These are repeated exploratory hospital evaluations,
not a new confirmatory lockbox. No thresholds changed.

No full-body neural retraining, calibration, runtime update or promotion.
Private derived features, weights and predictions remain outside Git.
Evidence: `artifacts/map_derived_gap_training_20260912.json`.
