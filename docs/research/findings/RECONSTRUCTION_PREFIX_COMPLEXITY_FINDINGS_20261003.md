# Earlier Readout Stages: A Partial Harm Reduction, Not a General Repair

## Completed Intervention

All 47 targets, five original folds, six stages (1/5/10/25/50/100) and four
saved readout families were replayed without refitting or changing the shared
encoder. Each held source component received a stage chosen by equal-component
MAE on OTHER source components. No held component's truth selected its stage.
Native, complete100-stage, population, process and trained-shuffle controls were
retained. sklearn staged_predict exactly reproduced all original endpoints.

Protocol: `RECONSTRUCTION_PREFIX_COMPLEXITY_PROTOCOL_20261003.md`.
Report: `artifacts/native_reconstruction_nonlinear_prefix_20261003.json`.
There are 1,797,884 query occurrences and 1,288 supported component decisions
per readout arm. Twenty-three unsupported decisions remain explicit, covering
24,701 occurrences: anion_gap17,376, myoglobin785 and pH6,540. Forty-four targets
have complete decision support. Unsupported cases use100-stage bookkeeping,
never a qualification or silently excluded row.

Twenty focused tests pass. Full original prediction/fit-curve endpoint parity,
role separation, all-target support and aggregate report bindings passed.
Patient rows, stage choices and feature caches stay local; only aggregates are
versioned. No outer-label assessment or new model training occurred.

## Result

No target passes the full five-comparison source diagnostic. In particular,
zero pass the corrected patient AND component comparison against the existing
100-stage body readout. Merely beating native is not incremental evidence.

| All47 target-macro normalized MAE | Body100 | Selected body prefix |
|---|---:|---:|
| Patient equal | 0.857139 | 0.856416 |
| Component equal | 0.811241 | 0.811243 |

The fully supported44-target subset has patient0.875882->0.875093 and
component0.831573->0.831566. There is no substantial macro repair. Across the45
targets with any supported decisions, the target-macro patient fraction using
a shorter body head is43.69%. The policy did act; failure is not an unopened gate.

Seven targets have numerically lower MAE at both levels than body100 (excluding
floating-point equality): CPK, LDH, lipase, respiratory rate, triglycerides,
troponin I and TSH. None clears the full corrected screen. Nine beat native
under that comparison's screen, but not body100; these are not nine promotions.

## Troponin Localizes a Real Conditional Tradeoff

| Troponin I normalized MAE | Native | Body100 | Selected body prefix |
|---|---:|---:|---:|
| Patient equal | 5.184358 | 5.226455 | 5.177608 |
| Component equal | 3.645206 | 3.699823 | 3.641345 |

Compared with body100, selected prefixes reduce patient MAE by0.048847 and
component MAE by0.058478. Conditional95%CIs are [-0.062751,-0.035714] and
[-0.101761,-0.019198]. However component one-sided p=0.012208 does not meet
the prespecified .05/(47*5) screen. Compared with native, the component CI
[-0.020396,0.016492] crosses0. The selected head also does not establish
increment beyond population, process-prefix and matched shuffle-prefix controls.
Do not describe harm reduction as validated individual physiological inference.

Short stages1/5 account for75.67% of patient-weighted choices and stage100 for0%.
The retrospective curves explain a failure mechanism:

| Fold-mean body-minus-native central-band MAE | Stage10 | Stage25 | Stage100 |
|---|---:|---:|---:|
| Fitting | -0.033917 | +0.009412 | +0.049840 |
| Source selection | -0.001980 | +0.062357 | +0.173258 |

All five fits improve this band at stages1/5/10, but all harm it at100. Meanwhile
tail error improves as stages accumulate. The aggregate objective permits a
conditional tradeoff already within fitting, then central harm grows further
on unseen groups. The first mean sign change lies between evaluated stages10
and25; this does not identify an exact intervening iteration or prove that a
different loss will generalize.

Central means absolute fit-normalized truth<=1, NOT healthy/stable patients.
These bands use the answer and cannot choose a deployed head or stopping time.
The selected-prefix result above uses other components, NOT a truth-band oracle.

## Why This Does Not Repair Every Target

| Patient-equal normalized MAE | Native | Body100 | Selected prefix |
|---|---:|---:|---:|
| Ferritin | 0.827030 | 0.845038 | 0.849749 |
| O2 saturation | 0.600807 | 0.602190 | 0.602363 |
| BUN | 0.635639 | 0.614053 | 0.614896 |
| INR | 0.893784 | 0.860433 | 0.860548 |

Ferritin nearly always shortens (only0.033% body100), yet remains worse than
native at both levels. Its component mean improves versus body100 but its
patient mean worsens. Shortening trades central-region recovery for lost gains
elsewhere; it is not a uniform overfitting cure.

BUN and INR mostly choose50/100. Shortening slightly worsens both means,
particularly INR's component error. Their positive average improvement over
native should not be discarded because a universal stopping rule fails.
O2 saturation also has mixed patient/component directions.

## Interpretation and Decision

Confirmed intervention: limiting later stages can reduce the adverse troponin
readout behavior. Not confirmed: a transferable, control-beating repair across
all groups or all targets. Replacing all readouts with earlier stages would
discard real gains and does not solve the full shared-state problem.

This refines the previous fit-gap conclusion: some late-stage conditional harm
is avoidable, but its avoidance is not enough to identify better-than-baseline
individual values in unseen groups. Ferritin has a different unresolved
transport/conditional tradeoff. Do not repeat generic early stopping as an
untested universal solution or infer that more capacity is required.

All statistics are conditional on overlapping original fits and reused source
data. Bootstrap intervals do not include repeated stage-selection fitting;
they are not untouched external evidence or formal runtime gates. This remains
masked CURRENT reconstruction, not forecasting or causal physiology. No release
weights, registry, calibrated intervals or serving behavior changed.
