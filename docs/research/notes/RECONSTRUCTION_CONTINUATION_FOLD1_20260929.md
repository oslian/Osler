# Fold 1 Fixed Monitored Continuation

Both real and process arms completed the prespecified epoch-10 endpoint.
No code, loss, sampling budget or selection rule changed. The epoch-8 fixed
6,862-query monitor reproduced before training. Each extra epoch uses 50,384
updates across all 47 targets. Epoch-10 paired sampling RNG, torch RNG, pending
queues and seen sets match exactly: 358,799 unique fit pairs.

## Fixed development monitor

Equal-target macro averages, fit-normalized units:

| Arm / epoch | Patient MAE | Component MAE | Smooth L1 |
|---|---:|---:|---:|
| Real 8 | 0.910917 | 0.903163 | 0.607903 |
| Real 9 | 0.915314 | 0.906881 | 0.608835 |
| Real 10 | 0.908001 | 0.900541 | 0.602059 |
| Process 8 | 0.973157 | 0.967397 | 0.655726 |
| Process 9 | 0.970275 | 0.965405 | 0.652970 |
| Process 10 | 0.973328 | 0.967953 | 0.655133 |

Real patient MAE improves 24/47 targets; component MAE improves 22/47. Process
improves 20/47 and 19/47 respectively. Real macro patient improvement is about
0.32%, component about 0.29%. Aggregate improvement does not imply most targets
improve at both weighting levels. Epoch 9 worsens while epoch 10 improves;
the endpoint remains the previously fixed epoch 10, not retrospective selection.

## Limits and next required evidence

This is masked current-value reconstruction on a fixed source-development
monitor, not future forecasting or a promotion test. Folds 0 and 1 overlap in
training/monitor populations; their gains are not independent replications.
Their small aggregate improvements warrant completing the fixed comparison,
not claiming universal repair or convergence.

Independent continuation folds 3 and 4 remain, both arms. Fold 2 reuse requires
explicit equivalence certification. Complete paired assessment still remains.
Private checkpoints and monitor JSONs: `monitored_continuation_20260929` under
the existing additional-development data root. Original epoch-8 weights remain
untouched. No runtime, registry, interval, promotion or GitHub release changed.
