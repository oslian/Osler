# MAP Unrelated Label Selection

## Located Scope Defect

The bedside label builder first restricted queries by `labels.npz['paired'][:,0]`,
inherited from the potassium task, before constructing MAP labels. MAP therefore
depended on another analyte having a future pair. This is outcome-dependent
cohort selection, not a future value passed into the predictor. Prior matched
comparisons remain comparisons on that restricted cohort; they do not establish
general MAP performance.

## Rebuild

Use all55,468existing queries and qualified MAP events, with exactly the same
current-age<=6h and future6h+/-1h policy. Old overlap current, age, future,
actual horizon, eligibility and future source IDs reproduce exactly.

| Scope | Pairs | Patients | Stays |
|---|---:|---:|---:|
| Old eligible | 3,925 | 1,303 | 1,442 |
| Recovered only | 29,552 | 1,438 | 1,583 |
| All MAP eligible | 33,477 | 1,505 | 1,653 |

Patient sets overlap. More rows do not mean proportionally more independent
patients. Original partitions retained; no test patient is moved to training.
Some original query patients have no qualified bedside MAP data. The original
query/cohort extraction remains restricted and is not MAP-native or full eICU.

## Remaining Work and Limits

New private label cache `map_all_query_labels.npz` removes this unnecessary
future-pair prerequisite. Full legal input history must be rebuilt for these
queries before retraining. Compare old and expanded training on identical
heldout MAP labels, with patient-equal weighting, and separately report newly
recovered rows. More support is not yet an accuracy repair or root-cause proof.
No training/runtime/calibration/promotion was performed.

## Input Rebuild Completed

`scripts/build_map_all_query_inputs.py` rebuilt all55,468 query snapshots and
fixed-lag histories. All old query AND history input/current/mask/age/presence/
horizon arrays reproduce bit-exactly. Eligible MAP current values agree with
the new labels after inverse scaling. New private `map_all_query_inputs.npz`
supports33,477pairs. Native ragged-event histories have not been expanded yet;
this is a ready fixed-lag input cache, not completed neural retraining.
Evidence: `artifacts/map_all_query_inputs_20260912.json`.

## Native History Rebuild Completed

`scripts/build_map_all_native_tokens.py` completed all55,468query histories:
427,180unique event groups, median96groups, maximum385. No truncation.
All5,526old query histories reproduce their complete token and treatment arrays
exactly, including NaNs in missing event values. The same availability,
source-version and conflict logic is used, with qualified bedside events.
New private `map_all_native_tokens.npz` is ready for matched neural training;
training itself remains pending. Evidence:
`artifacts/map_all_native_tokens_20260912.json`. No runtime/promotion changes.

## Source Transform Equivalence

`audit_map_expanded_transforms.py` checked all seven original fitting subsets
under old and expanded native cache layouts. Allowed channels, numeric centers
and scales, and event centers/scales are bit-exact. Event reindexing therefore
does not itself alter these source transforms. This is a preprocessing
equivalence check, not a neural training or accuracy result. Evidence:
`artifacts/map_expanded_transforms_20260912.json`.

Evidence: `artifacts/map_label_selection_20260912.json`,
`scripts/audit_map_label_selection.py`. Aggregate-only repository output.
