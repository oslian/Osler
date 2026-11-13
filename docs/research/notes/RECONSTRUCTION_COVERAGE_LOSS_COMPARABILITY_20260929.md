# Training Loss Is Not a Fixed-Cohort Learning Curve

Replay the exact eight-epoch schedules of all four independent coverage fits.
Final RNG, pending queues, seen sets and per-epoch unique counts reproduce the
saved checkpoints exactly. At each draw compute SmoothL1(beta1) of a FIXED
fit-only median predictor. No model training, selection or outer data used.

| Fold | Constant loss range | Training loss range | Epoch correlation |
| --- | ---: | ---: | ---: |
| 0 | .084599 | .102857 | .780819 |
| 1 | .055510 | .073846 | .698640 |
| 3 | .033623 | .076713 | .766096 |
| 4 | .087040 | .091774 | .798661 |

The constant predictor cannot learn, so its fluctuations come from changing
draw composition. This establishes that current online epoch losses do not
identify convergence. Correlation across8points is descriptive; it does not
partition causally how much model loss change is due to sampling. Even
baseline-relative online loss uses weights changing during the epoch, so it
cannot establish a plateau or diagnose generalization.

## Repair Required for the Next Training Experiment

Keep the completed fixed-eight-epoch comparison immutable. For any continuation,
declare a fixed source-only monitor cohort before updates, retain its exact
query identities and evaluate at fixed checkpoints. Report per-target and
component errors on the same examples, alongside the online optimization loss.
Do not use outer scores to choose epoch. Keep checkpoint snapshots or complete
monitor outputs; the current trainer overwrites prior model states, so a true
retrospective fixed-query learning curve cannot be reconstructed from logs alone.
This is a monitoring gap, not proof that more epochs will repair the model.

Artifact:`artifacts/native_reconstruction_coverage_loss_comparability_20260929.json`.
Reproduce:`python -m scripts.audit_coverage_loss_comparability`.
Unit test verifies NumPy baseline loss exactly matches the PyTorch training
objective. No trained weights, runtime or promotion changed.
