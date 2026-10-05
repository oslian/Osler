# MAP Availability-Native Raw History Trial

## Fixed Comparison

Same MAP-native labels and original patient partitions for own MAP history,
full raw history, and observation-support control. The latter keeps own MAP
numbers and all available masks/ages, but zeroes other numerical values.
Only source-fit patients determine supported variables. Fixed HGB absolute-error
loss, 200 iterations, 15 leaves, learning rate .05, minimum leaf 30, L2 10,
63 bins, no early stopping or tune selection. This uses the established tree
family but is not identical to prior leaf-grid-selected training.

All eligible rows retained. Fit weights and MAE give patients equal weight.
Source evaluation has 295-329 patients across seven overlapping splits;
held-out hospitals have 185 patients/11,071 rows in every split. Scores and
paired 20,000-draw 95% patient bootstrap use identical rows for all models.

## Completed Results

Median patient-equal MAE across seven splits, mmHg:

| Method | Source evaluation | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.44082 | 11.73923 |
| Own MAP history | 8.78353 | 10.24211 |
| Full raw history | 8.74362 | 10.37322 |
| Observation-support control | 8.73532 | 10.36161 |

Full versus own: 0/7 significant gains, 1/7 harms in both roles.
Full versus support: 0/7 gains or harms in both roles.
Full versus persistence: 7/7 gains, no harms in both roles.
All 21 fits completed; 41 direct artifact bindings verified. Three focused
tests pass, including feature-mode isolation and patient/event counting.

## Interpretation and Limits

Changing the query task to MAP availability does not establish stable extra
cross-variable value for this fixed-lag readout. It is not enough to repair the
incremental full-context gap. Because this trial bypasses the encoder, JEPA
compression alone cannot explain every negative comparison in the project.
Neither point is proof that all other physiological information is absent.

Do not compare old/new MAEs as paired repair effects: anchors, outcome reuse,
patient composition, and model selection differ. The query universe still
inherits earlier stay selection, not all eICU. Native event sequences and the
complete persistent anatomical architecture were not trained here. More than
one anchor may share future truth; patient clustering limits row pseudoreplication
but does not make repeated outcomes independent. Seven splits are not seven
independent hospital cohorts. Support/no-number equivalence is not established
by nonsignificance. No new database, conformal audit or full promotion gate.

No release weight, registry, calibrator or runtime changed. Status remains
diagnostic_only, causal_claim_allowed remains false.
Evidence: `artifacts/map_native_anchor_tree_20260912.json`; private predictions,
source models and input feature cache stay outside version control.
