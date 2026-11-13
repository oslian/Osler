# MAP Native Full-Context Increment Shrinkage

## Fixed Repair Test

Keep the trained own-history forecast as anchor, and use
`own + alpha * (full - own)`. Select alpha from0,.1,...,1 using only original
source-tune patient-equal MAE; no hospital selection. Same existing 21 fitted
trees and MAP-native labels. This reuses the earlier magnesium diagnostic
pattern on a different target/task; it is not a new neural architecture.

Selected alphas across7splits: .4,.6,.9,.5,.7,.1,.6.

| Method | Source evaluation MAE | Hospital MAE |
|---|---:|---:|
| Own | 8.78353 | 10.24211 |
| Full | 8.74362 | 10.37322 |
| Selected increment | 8.72261 | 10.28760 |

Patient-paired20k bootstrap,95% CI:

| Comparison | Source gain/harm | Hospital gain/harm |
|---|---:|---:|
| Selected vs own | 4/7 /0 | 0/7 /0 |
| Selected vs full | 3/7 /0 | 5/7 /0 |
| Selected vs persistence | 7/7 /0 | 7/7 /0 |

## Decision

This is a partial repair: attenuating the extra context improves the original
full model and sometimes the own model inside source. It does not establish
cross-hospital incremental value beyond own history. The latter remains
numerically better in median hospital MAE. No complete promotion, calibrated
interval, new database, neural retraining or runtime change.

The result supports testing over-strong full-context corrections, but not a
unique variance/bias/biological cause. Scalar alpha cannot discover which future
patients benefit. Tune selection and reused evaluation limit confirmatory
claims; do not select another alpha from hospital scores. Evidence:
`artifacts/map_native_increment_shrink_20260912.json`.
