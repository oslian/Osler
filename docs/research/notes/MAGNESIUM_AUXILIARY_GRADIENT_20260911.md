# Actual auxiliary gradient geometry

Seven selected real-auxiliary checkpoints, three fixed source-fit batches each.
Reconstructed native inputs and verified recorded event scalers. No optimizer
updates. Compared primary Mg value+JEPA gradient against the actual weighted
.25 auxiliary value+JEPA gradient on shared encoder/filter/mixer parameters.
The reused geometry report's `retention_*` field names refer to this real-label
auxiliary gradient, NOT initial-output retention or distillation.

| Shared block | Median auxiliary/primary norm | Median cosine | Opposed /21 |
|---|---:|---:|---:|
| Combined | .1766 | .0105 | 10 |
| Belief filter | .1692 | -.1105 | 13 |
| Variable mixer | .1659 | -.1455 | 15 |
| Module mixer | .1528 | -.0775 | 12 |
| State encoder | .1756 | .0285 | 9 |

The auxiliary signal is not literally absent. Its average objective coefficient
cannot be equated to a hundredfold smaller gradient. Near-orthogonal and mixed
sign directions mean increasing its weight also increases conflicting updates.
This is local geometry at selected checkpoints, not the whole optimization
trajectory, a statistical causal diagnosis, or proof that gradient surgery will
repair generalization. It does not explain the primary-only whole-own gap.
Earlier retention-gradient work used a different objective and remains distinct.

No training, runtime change, calibration, promotion or new lockbox.
Artifact: `artifacts/magnesium_auxiliary_gradient_20260911.json`.
