# Record-anchored prediction: partial increment, no stable repair

Twenty-one fixed fits complete on rebuilt5526pairs; patient roles unchanged.
Median patient-equal MAE:

| Role/group | Real treatment | Support only | Own | Persistence |
|---|---:|---:|---:|---:|
| Source/all | .401521 | .409454 | .409454 | .496675 |
| Source/bracket-after | .409416 | .411906 | .411906 | .508829 |
| Hospital/all | .393562 | .402921 | .402921 | .482238 |
| Hospital/bracket-after | .406434 | .422643 | .422643 | .523661 |

Real-vs-own gain1/7source both groups,0/7hospital all,2/7hospital bracket-after.
Real-vs-support gain1/7every group. No significant harm in these comparisons.
Real-vs-persistence7/7every group. Hospital bracket group146patients.
Own/support median equality is not prediction identity; bootstrap differences
must not be inferred by subtracting separately computed medians.

Partial positive treatment increment, not seven-seed stability, complete gate,
calibrated interval or shared-JEPA improvement. New tasks have different anchors,
labels and effective patients; old-vs-new unmatched MAEs do not isolate repair.
The timing-contract restriction was real; this result does not prove it alone
caused prior failures. No online state updating or causal effects tested.

Report: `artifacts/rate_anchored_prediction_20260911.json`. No runtime change.
