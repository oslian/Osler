# Fold 3 Fixed Monitored Continuation

Both arms completed the prespecified epoch 10. Epoch-8 monitor parity passed;
7,167 fixed source-selection queries cover all 47 targets. Each extra epoch
uses 50,384 updates. Paired sampling RNG, torch RNG, pending queues and seen
sets match exactly at epoch 10: 350,083 unique fit pairs. No training settings
or selection rules changed; the original epoch-8 checkpoints remain untouched.

## Equal-target macro monitor results

Fit-normalized units; these are descriptive development metrics.

| Arm / epoch | Patient MAE | Component MAE | Smooth L1 |
|---|---:|---:|---:|
| Real 8 | 0.893425 | 0.829798 | 0.582825 |
| Real 9 | 0.894739 | 0.824426 | 0.585357 |
| Real 10 | 0.892299 | 0.825874 | 0.579677 |
| Process 8 | 0.942637 | 0.876667 | 0.622405 |
| Process 9 | 0.945071 | 0.878755 | 0.624342 |
| Process 10 | 0.945477 | 0.878324 | 0.623546 |

Real patient MAE improves 26/47 targets and component MAE 28/47. Process
improves 19/47 and 17/47 respectively. Real aggregate improvements are about
0.13% patient and 0.47% component. Real component MAE is better at epoch 9
than epoch 10, but the endpoint remains the predeclared epoch 10.

The three completed fits have small real-model macro improvements, not
independent confirmation or uniform target gains. Monitor/fitting populations
overlap across folds. No significance or promotion claim is supported yet.

Remaining: fold 4 both arms, explicit fold-2 equivalence certification, and
complete paired assessment. This is masked current-value reconstruction,
not future forecasting. Runtime, intervals, registry and release are unchanged.
Private immutable checkpoints/monitor JSONs remain in
`monitored_continuation_20260929` under the additional-development data root.
