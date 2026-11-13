# Seed19 Replication: Completed Training And Heldout Evaluation

Training: `artifacts/native_reconstruction_replication_v2_19_epoch8_20260914.json`.
Evaluation: `artifacts/native_reconstruction_replication_v2_19_evaluation_20260914.json`.

All three corrected arms completed eight epochs, 293,904 updates each. Every
epoch consumed all 26,264 eligible fit patient-target pairs. Checkpoint/source
hashes, selected epochs and report values were verified after completion.
The invalid V1 process branch was not used. Five partition/wiring tests pass.

| Arm | Selected epoch | Source tune normalized MAE |
|---|---:|---:|
| Real numeric inputs | 7 | 1.772240 |
| Observation-process only | 4 | 1.873076 |
| Shuffled labels | 1 | 1.914907 |

Common initial source score: 1.914917. Source selection is not heldout evidence.
The evaluator rebuilds seed19 patient roles, ignores old shard roles, refits
population medians using fit patients only and checks unique patient-target
pairs. It evaluates 14,064 heldout pairs across all 39 source-eligible targets.
No seed7 fitted predictions or weights enter this evaluation.

## Nominal Paired Results

Real beats process on 18 evaluation and 16 hospital targets. Nine overlap:
albumin, ALT, AST, bicarbonate, calcium, creatinine, phosphate, total protein,
WBC. Requiring real to beat population AND process AND placebo in BOTH roles
leaves seven: albumin, AST, bicarbonate, calcium, creatinine, phosphate,
total protein.

Compared with the previous seed7 nominal results, four satisfy the combined
comparisons in both runs: albumin, AST, bicarbonate, total protein. This is
descriptive repeated-split consistency, not four promoted targets. The same
hospital patients are reused; patient cohorts overlap between splits, and
initialization/sampling also change with seed. These are not independent studies.

Creatinine's new gain did not reproduce a prior stable result: seed7 had no
significant benefit over process in both roles. Heart rate and platelets lost
their two-role process advantage in seed19. MAP significantly harms the
evaluation population baseline. Do not select only the successful split.

Unsupported paired intervals remain explicit: evaluation anion gap has 25
patients; hospital fibrinogen/LDH/TSH have 26/27/17. No minimum-support change.

## Diagnosis And Remaining Work

Corrected complete training can recover incremental numerical information in
the shared state; universal empty-latent claims are contradicted by these
results. Generalization is still target- and split-dependent. Process-only
prediction remains a strong comparator, and all intervals here are nominal,
not multiplicity-adjusted. This does not resolve the earlier multiplicity
sensitivity concern or establish a physiological causal mechanism.

Remaining predefined seeds, multiplicity sensitivity, process-shift and truly
untouched external validation remain unfinished. This is current-value masked
reconstruction, not future forecast or conformal interval validation. No runtime
weights, registry, interval or serving changes; no promotion.
