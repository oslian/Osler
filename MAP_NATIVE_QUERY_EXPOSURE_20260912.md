# Native MAP Query Training Exposure

Replayed the exact RNG sequence and patient/query sampler in the completed
native-query runner. A unit test matches the runner's list-comprehension draws
at every epoch. Patient pools and seed-specific selected epochs come from the
bound training data/reports. No new training or selection is performed.

At epoch 32, unique fit query coverage is 10.07-10.77%. Selected full checkpoints
have 6.58-10.75% coverage. This is query coverage, not patient or unique future
measurement coverage. It does not by itself prove inadequate training.

Selected-checkpoint comparisons, same query rows for both arms:

| Fit-query exposure | Median paired full-own MAE | Significant gain/harm |
|---|---:|---:|
| Both checkpoints saw the query | +0.22127 | 0/7 gain, 6/7 harm |
| Neither checkpoint saw the query | +0.29949 | 0/7 gain, 7/7 harm |

Exclusive exposure is also recorded. These groups exactly partition fit rows.
The harm is therefore not confined to unvisited queries. Low coverage remains
a potential training limitation but is not an adequate sole explanation. This
is MAE at tune-selected checkpoints, not the per-step total JEPA loss, and no
convergence proof. Seen/unseen queries can share patients and future outcomes;
these are fitting diagnostics, not independent generalization gates. Seven
splits use overlapping patients, not seven independent cohorts.

No runtime/calibration/promotion or claim that extra training cannot help.
Evidence: `artifacts/map_native_query_exposure_20260912.json`.
