# Matched magnesium-primary input comparison

Seven whole-input fits completed. Saved own controls and all bound files were
verified. Same eligible patients, partitions, primary loss, optimization budget,
initialization, and magnesium source-tune selection; input scope differs.

| Median patient-equal MAE | Own history | Whole input | Persistence |
|---|---:|---:|---:|
| Source evaluation | .217013 | .218283 | .277679 |
| Hospital heldout | .182700 | .184960 | .238434 |

Whole vs own: significant improvement0/7 and harm0/7 in both roles.
Both beat persistence7/7 in both roles. Selection epochs5,11,14,17,23,25,5.
Thus glucose-primary versus magnesium-primary selection was not a sufficient
explanation of the previous missing whole-context increment. This does not
establish equivalence or absence of useful cross-variable information.
Additional context, representation, optimization and limited support remain
possible contributors; this experiment does not isolate them individually.

Overlapping seven splits are not independent replications. This is reused eICU
evidence on potassium-selected queries, not a new MIMIC/NWICU/lockbox result.
No calibration, anatomical persistent-model claim, runtime change or promotion.
Artifact: `artifacts/native_magnesium_matched_20260911.json`.
