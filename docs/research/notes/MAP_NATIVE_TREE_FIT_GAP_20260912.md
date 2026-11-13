# MAP Native-Anchor Fit Gap and Stopping

## Failure Location

Replayed all saved fixed-HGB predictions; original held-out MAEs and paired
comparisons reproduce exactly. Median patient-equal MAE:

| Role | Own | Full | Support |
|---|---:|---:|---:|
| Fit | 8.23184 | 7.92778 | 7.95890 |
| Tune | 8.92151 | 8.88831 | 8.90729 |
| New source patients | 8.78353 | 8.74362 | 8.73532 |
| Held-out hospitals | 10.24211 | 10.37322 | 10.36161 |

Full-own fit gain7/7, full-support fit gain4/7. Both incremental comparisons
gain0/7 on tune, new source and hospitals. Fit scores are resubstitution, not
independent evidence of useful physiology. The earliest measured loss of
stable incremental advantage is already at new source patients. Hospital
shift alone is therefore not a sufficient explanation for this trial.

## Matched Stopping Test

Used existing staged predictions, same source-tune-only grid0/25/50/100/150/200
for all three arms. Zero denotes persistence. No fitting or test-based stage
selection. Last-stage tune scores reproduce final predictions. Full selects
200 in6/7,150 in1/7; own200in5/7,150in2/7. Support200in4/7,150in3/7.

Selected full versus fixed200 full: gain0/7 both roles, harm1/7newsource,
0hospital. Selected full versus equivalently selected own: gain0/7both,
harm2/7newsource,1/7hospital. Median MAEs unchanged. Earlier stopping does
not repair the gap on this grid. Boundary selection does not prove convergence
or justify claiming universal overtraining; longer optimization is a distinct
question. Small in-sample advantages do not by themselves identify their cause.

## Limits

No new external lockbox, interval calibration, neural retraining or promotion.
No runtime weights/registry changed. Findings apply to this fixed-lag raw tree
diagnostic, not the complete anatomical body-state architecture. Reused
evaluation and overlapping splits preclude a new confirmatory claim.

Evidence: `artifacts/map_native_tree_fit_gap_20260912.json` and
`artifacts/map_native_tree_stopping_20260912.json`. Predictions remain private.
