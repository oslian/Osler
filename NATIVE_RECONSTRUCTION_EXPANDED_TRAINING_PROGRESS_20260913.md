# Expanded Training Progress

Protocol: `NATIVE_RECONSTRUCTION_EXPANDED_TRAINING_PROTOCOL_20260913.md`.
Epoch-one report: `artifacts/native_reconstruction_expanded_training_epoch1_20260913.json`.

Both arms completed the first of eight prespecified epochs. Each consumed all
25,511 source-eligible fit patient-target pairs, with 35,997 updates including
balanced repeats across 39 source-selected targets. The larger cache includes
317 additional fit pairs for source-ineligible targets; these were not trained.

| Arm | Initial normalized tune MAE | Epoch 1 | Selected epoch |
|---|---:|---:|---:|
| Real labels | 2.085252 | 2.011559 | 1 |
| Deranged fit labels | 2.085252 | 2.086760 | 0 |

The real model's approximately 3.5% source-tune improvement is preliminary,
not independent heldout evidence, statistical significance, or promotion.
Placebo preserves target-label marginals and capacity but breaks fit patient
alignment. Both use the real source-tune labels and may select initialization.
Evaluation/hospital prediction errors have not been inspected for this run.

Private checkpoints retain current and selected weights, optimizer state,
NumPy scheduling RNG and Torch RNG, guarded by dependency hashes. Continue
the same experiment with `--through-epoch 8`, rather than reinitializing.
Three schedule/derangement tests pass. No release/runtime/interval changes.
