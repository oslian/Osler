# MAP Support-Control Path Replay

Seven trained numeric-control checkpoints were replayed on source-tune only.
Original predictions reproduce exactly for every seed. No new weights or
external predictions were fitted/selected. Crossed history and query input
views separately use original support-control or own-MAP-only representation.

| Fixed-weight intervention | Median paired MAE change | Significant gain / harm |
|---|---:|---:|
| Own-only query; original history | +0.46694 | 0/7, 7/7 |
| Own-only history; original query | +0.07694 | 0/7, 3/7 |
| Both own-only | +0.65032 | 0/7, 7/7 |

The trained control is sensitive to its current query support/age/presence,
not merely the number of recurrent updates. Simply suppressing these inputs
after training is not a repair. Nor does removing non-MAP history updates
improve this fixed model. Relative intervention magnitudes are descriptive:
the script does not directly bootstrap their difference.

These are off-training-distribution interventions, not causal physiology,
proof that support features are beneficial, or a matched-training verdict.
In particular, inference input removal does not transform control weights
into the independently trained own model. The original own model's lower
error cannot be recovered simply by making its input layout look own-only.

This localizes reliance within the trained control, not the unique cause of
its worse accuracy. A training-matched query/history factorial remains needed
to distinguish harmful learned dependencies from input mismatch. Reused
source-tune results are diagnostic, not untouched validation. No runtime,
calibration or promotion changes.

Evidence: `artifacts/map_control_paths_20260912.json`.
