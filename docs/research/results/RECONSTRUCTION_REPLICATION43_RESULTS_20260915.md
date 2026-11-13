# Seed43 Reconstruction Replication

Completed the unchanged eight-epoch protocol and heldout evaluator:

- `artifacts/native_reconstruction_replication_v2_43_epoch8_20260914.json`
- `artifacts/native_reconstruction_replication_v2_43_evaluation_20260914.json`

The suffix is the frozen experiment identifier, not a claim that execution
finished on September14. All training/evaluation dependency hashes verify.
All39 targets meet source support. Each arm consumed25,552 unique fit pairs
per epoch,35,724 updates per epoch and285,792 updates total. Evaluation used
14,530 heldout patient-target pairs with split-specific role mapping and medians.

| Arm | Selected epoch | Source tune normalized MAE |
|---|---:|---:|
| Real | 7 | 1.715819 |
| Trained process-only | 4 | 1.855242 |
| Placebo | 0 | 1.888180 |

Initial score1.888180. Selected placebo is exactly the population median in this
split; these two reference comparisons are not independent confirmations.

## Heldout Results

Nominal95% paired intervals favor real over process on19evaluation and15hospital
targets, with ten in common. Requiring superiority over population, process and
placebo in both roles leaves nine: albumin, ALT, AST, calcium, heart rate,
hematocrit, hemoglobin, PTT, total protein.

Across completed seeds7/19/31/43, the intersection of the combined comparisons
shrinks to three: albumin, AST, total protein. Bicarbonate is no longer consistent.
No nominal significant harms against these comparators were found in this split;
that is not proof of equivalence or noninferiority for the other targets.
Hospital fibrinogen/LDH/TSH remain unsupported with26/27/17 patients.

## Important Counterexample

Bicarbonate evaluation real/process MAE0.835946/0.845822: paired difference
-0.009876, CI[-0.053966,0.033366]. Hospital real/process0.928480/0.927905:
difference+0.000575, CI[-0.066258,0.068144]. Hospital versus population also
fails: difference-0.044181, CI[-0.136635,0.048039]. These are normalized units.

This is not merely a strong gain missing a tiny significance margin: incremental
numerical utility over the process comparator is near zero in this fit. It does
not establish absence of biological information. Split membership, model
initialization, sampling and shared multi-target optimization all vary together.
Do not attribute the change uniquely to patient biology or lack of raw data.

Creatinine improves over process on hospital but its evaluation CI crosses zero
(-0.052078, CI[-0.109622,0.007244]); do not call its two earlier gains universal.

## Boundary

The same hospital patients are reused; these four splits are not independent
studies. Tests are nominal and multiplicity is not resolved. Current-value
masked reconstruction is not forecasting or calibrated interval validation.
Remaining fixed seeds59/71/83 are pending. No runtime changes or promotion.
