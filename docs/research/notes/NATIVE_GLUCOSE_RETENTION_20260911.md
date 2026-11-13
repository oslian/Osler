# Other-target retention after glucose-only training

New diagnostic of the same seven native whole checkpoints, not new training.
Matched initial/final identity decoder and native inputs; only glucose loss
selected the trained weights. Other-target labels were not used for selection.
Require >=30 patients per reported target/role. Patient-paired bootstrap2,000.
Aggregate: `artifacts/native_glucose_retention_20260911_v2.json`.
V2 adds parameter drift; all V1 scores exactly reproduce and bindings verify.

Selected hospital findings (MAE in each target's native unit):

| Target | Initial | Glucose-trained | Persistence | Significant harm / 7 |
|---|---:|---:|---:|---:|
| Calcium | 0.5072 | 0.6294 | 0.3825 | 6 |
| Magnesium | 0.2074 | 0.3168 | 0.2384 | 7 |
| Phosphate | 0.9907 | 1.5838 | 0.9973 | 7 |
| Hematocrit | 3.5940 | 4.3128 | 3.3001 | 6 |
| Potassium | 0.4391 | 0.4765 | 0.4821 | 5 |

Not every target worsens: BUN7.2587->5.2409 and creatinine0.3928->0.3092
improve7/7 hospital comparisons, but both remain numerically worse than
persistence4.1766/0.2512. Primary glucose improves46.0267->42.4007.
New-source magnesium harms7/7, calcium/hematocrit6/7. Full table in aggregate.

## Mechanism evidence

All208 non-glucose value-head tensors are exactly unchanged in all seven fits.
Shared state-encoder relative L2 change1.22%-4.85%, variable mixer0.60%-3.05%,
module mixer0.59%-3.33%. Other shared prediction components can also change;
these numbers do not isolate one layer. Glucose-only adaptation changes the
inputs to unchanged other readouts, without a retention objective for those
targets. Functional regression is measured, not inferred from parameter drift.

This identifies a multi-target retention failure in the candidate procedure.
It does NOT establish this is the cause of glucose whole-vs-own failure, nor
that formerly deployed experts regressed: runtime was never replaced, and these
other heads were not separately promoted under the native identity contract.
Multiple unadjusted target comparisons and overlapping seeds are exploratory;
no overall-family statistical claim or newly promoted BUN/creatinine is made.

The repair direction is to protect validated target behavior while updating the
shared pathway, assessed with this retention control alongside primary gains.
Prior auxiliary truth supervision already failed stable primary gain on an
older task; do not simply repeat it as an untested universal solution. Matching
native-task retention constraints and measuring their tradeoff remains distinct.
No causal claims, calibration, runtime changes or new untouched database.
