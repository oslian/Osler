# JEPA Input Provenance Findings

Date: 2026-07-13

## Question

Can the JEPA real-data failure be partly caused by the state extraction path
turning missing values into apparent physiology, while the hand-built belief
layer removes that noise first?

## Controlled A/B test

The same `dka_symbolic_jepa_v5.pt` checkpoint and the same
`dka_transitions_6h_demo_v4.parquet` cohort were evaluated twice. No retraining,
checkpoint change, action change, or target change was performed.

- `legacy`: historical evaluator passes the full derived/default `s2vec(state)`.
- `measurement_pure`: only values present in the observed state are passed as
  state values. Missing dimensions are zeroed in the normalized state vector;
  observation mask and measurement age remain available to the model.

The measurement-pure path matches the training contract in
`train_intervention_jepa.py`, where the input is `state * mask`.

## Results

Active-DKA rows, same rows in both modes:

| Target | Persistence | Legacy JEPA | Measurement-pure JEPA | Change |
| --- | ---: | ---: | ---: | ---: |
| Glucose | 96.647 | 148.321 | 123.928 | -24.394 |
| Bicarbonate | 2.750 | 5.221 | 2.660 | -2.561 |
| Anion gap | 3.250 | 3.039 | 3.102 | +0.063 |
| Potassium | 0.440 | 0.690 | 0.681 | -0.009 |
| MAP | 16.268 | 13.036 | 13.667 | +0.631 |
| Sodium | 4.158 | 9.830 | 10.059 | +0.229 |
| Osmolality | 9.534 | 22.122 | 21.516 | -0.606 |
| Creatinine | 0.340 | 0.482 | 0.483 | +0.001 |
| Urine output | 53.787 | 144.971 | 112.964 | -32.008 |

The cohort is only 12 stays and 42 active-DKA comparisons, so this is a
diagnostic ablation, not a promotion result.

The same A/B was then repeated on the full MIMIC-IV DKA transition artifact:
755 subjects, 1,010 stays, 10,785 rows, and 3,576 active-DKA comparisons.
Active-DKA results were:

| Target | Persistence | Legacy JEPA | Measurement-pure JEPA |
| --- | ---: | ---: | ---: |
| Glucose | 143.374 | 175.731 | 153.898 |
| Bicarbonate | 3.570 | 4.122 | 3.361 |
| Anion gap | 5.024 | 5.452 | 4.119 |
| Potassium | 0.595 | 0.840 | 0.855 |
| MAP | 11.523 | 11.542 | 11.666 |
| Creatinine | 0.265 | 0.679 | 0.646 |
| Urine output | 81.300 | 140.584 | 121.651 |

The correction is therefore real and reproducible, but it does not by itself
make the simulator-trained JEPA competitive on every target.

## History ablation

The existing `sequence_worldmodel_vs_ridge.py` was also run on the same full
MIMIC-IV cohort. It compared persistence, snapshot-plus-hand-belief ridge,
GRU history, differentiable Holt belief, and VICReg history. The hand-belief
ridge was best for glucose (65.266 MAE), sodium (2.427), bicarbonate (2.008),
and heart rate (7.911). GRU was best for anion gap (2.568) and lactate
(0.933), while VICReg was best for potassium (0.420) and MAP (9.660).

This supports a second diagnosis: the manual belief layer exposes a useful
patient-specific level/slope representation, while v5's history input is
primarily treatment history. A generic sequence learner does not automatically
recover the hand-crafted state, so the next JEPA candidate should add
measurement history and target-specific robust filtering rather than only
increase latent size.

## Interpretation

The extraction hypothesis is supported, but not sufficient:

1. Glucose, bicarbonate, urine output, pH, and osmolality improve when hidden
   defaults and derived values are not presented as observed state.
2. The improvement does not consistently beat persistence. Potassium,
   sodium, creatinine, and MAP do not show a uniform gain.
3. The hand-built belief layer likely wins partly because it performs explicit
   target-specific filtering, temporal isolation, and per-patient updating.
4. JEPA still has independent bottlenecks: simulator-to-real mismatch, a
   shared multi-target objective, sparse/irregular labels, and incomplete or
   coarse treatment-history representation.

## Safety boundary

The corrected `measurement_pure` path is now the default for new evaluations;
`legacy` remains available only for historical reproduction. This is a factual
observational evaluation only. It does not identify causal
treatment effects, does not validate hidden-state truth, and does not promote a
checkpoint. The legacy result is retained for reproducibility; the
measurement-pure path is a candidate evaluation correction.

## Next experiment

Before retraining, repeat this A/B on a larger patient-held-out cohort and add a
target-specific extraction ladder:

1. direct measured value only;
2. measured value plus age/slope/history;
3. derived value as a separate provenance channel;
4. default value only through mask-aware imputation.

Only after this audit is stable should a JEPA retrain use the corrected input
contract. The hand-built beliefs remain the reference for signal-isolation
quality, not a causal oracle.
