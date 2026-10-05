# Fold 0 Fixed Monitored Continuation

## Question and fixed comparison

Continue both real-value and process-control models from epoch 8 to the
prespecified epoch 10. Same 47 targets, sampling budget, fit patients, optimizer
and fixed 6,862-query source-selection monitor. This is current-value masked
reconstruction, not future prediction. No best-epoch selection.

## Verified execution

Both arms completed 50,384 updates per additional epoch. At epoch 10 their
sampling RNG, torch RNG, pending queues and seen-pair sets match exactly;
358,828 unique fit pairs have been seen. Epoch-8 monitor parity passed before
continuation. Original eight-epoch checkpoints were not overwritten.

Private checkpoints and monitor reports are in
`data/research_recovery/additional_development_domains_20260921/monitored_continuation_20260929/`.

## Fixed monitor results

All metrics below are equal-target macro averages in fit-normalized units.

| Arm / epoch | Patient MAE | Component MAE | Smooth L1 |
|---|---:|---:|---:|
| Real 8 | 0.911950 | 0.906847 | 0.606270 |
| Real 9 | 0.909960 | 0.907090 | 0.604332 |
| Real 10 | 0.908737 | 0.904325 | 0.601633 |
| Process 8 | 0.961297 | 0.959165 | 0.648717 |
| Process 9 | 0.965500 | 0.963469 | 0.650282 |
| Process 10 | 0.965707 | 0.964630 | 0.647745 |

Real epoch 10 improves patient MAE for 29/47 targets and component MAE for
28/47 versus epoch 8. Process improves each MAE for 25/47 despite worse macro
MAE. Therefore neither arm improves every target. Real macro patient MAE
improves about 0.35%; component MAE about 0.28%.

## Interpretation and remaining work

The monitored real/process difference grows in this fold. This is descriptive
evidence worth testing across the remaining fixed folds, not a significance
test or proof of a universal physiology/generalization repair. Process Smooth
L1 decreases while its MAE increases, so the training objective and MAE cannot
be treated as interchangeable convergence evidence.

Remaining independent continuations: folds 1, 3 and 4, both arms; fold 2 reuse
requires explicit certification. Complete paired evaluation remains necessary.
No thresholds, roles, runtime registry, intervals or promotion changed. The
published epoch-10 real weights remain candidate-only. All 47 targets remain
in scope, including those that worsened.
