# Source-selected whole-context increment shrinkage

Fixed grid0,.1,...,1 selected solely by patient-equal source-tune MAE.
Prediction own + alpha*(whole-own). Same saved verified matched Mg weights,
labels and partitions. No neural retraining or external parameter selection.
Selected alpha .1,.5,.8,.2,.3,.1,.2. Zero would count as rejecting increment.

| Median MAE | Own | Whole | Selected blend |
|---|---:|---:|---:|
| Source evaluation | .217013 | .218283 | .215612 |
| Hospital | .182700 | .184960 | .182648 |

Blend vs own significant gain0/7harm0/7both. Vs whole sourcegain2/7,
hospital0/7, no significant harm. Persistence7/7both. Shrinking helps some
source comparisons but does not establish robust extra-context value. Not
evidence that every patient's direction is correct or a universal amplitude
root cause. Tune reused for checkpoint and blend selection, not an independent
calibration/lockbox. Overlapping seeds are not independent replications.
No runtime, intervals or promotion.
Artifact: `artifacts/magnesium_increment_shrink_20260911.json`.
