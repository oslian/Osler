# Seed59 Reconstruction Replication

Completed the fixed eight-epoch training and heldout evaluation on 2026-09-16.
Reports retain their frozen experiment suffix:

- `artifacts/native_reconstruction_replication_v2_59_epoch8_20260914.json`
- `artifacts/native_reconstruction_replication_v2_59_evaluation_20260914.json`

All training/evaluation bound files passed SHA256 verification. Each of three
arms consumed26,254 unique fit patient-target pairs per epoch,36,176 updates
per epoch (289,408 total). Anion gap has insufficient source tune support26
and remains explicitly unsupported;38targets trained. No seed replacement.

| Arm | Selected epoch | Source tune normalized MAE |
|---|---:|---:|
| Real | 8 | 1.978913 |
| Trained process-only | 8 | 2.090744 |
| Placebo | 0 | 2.145206 |

Initial score2.145206. Both real and process select the last allowed epoch;
complete prescribed exposure is not evidence of optimizer convergence. Do not
extend only this seed based on its heldout scores. Selected placebo is exactly
the population baseline and is not an independent confirmation of that baseline.

## Heldout Comparison

Evaluated13,806 pairs with seed59 roles and fit-only population medians.
Nominal paired95% intervals favor real over process on16evaluation and9hospital
targets, six overlapping: albumin, alkaline phosphatase, ALT, AST, MAP,
total protein. MAP fails the combined comparisons against population/placebo
in both roles. Five satisfy all three references in both roles: albumin,
alkaline phosphatase, ALT, AST, total protein.

The intersection across completed seeds7/19/31/43/59 remains albumin, AST,
total protein. These are nominal repeated-split results, not five independent
studies or a final promotion count. All splits reuse the same hospital group.
No nominal significant harms against the three references were found in this
split; this does not prove equivalence for unselected targets.

Hospital fibrinogen/LDH/TSH still have26/27/17 patients, below the fixed minimum.
The original39-target catalog is preserved through explicit unsupported status.

## Interpretation And Remaining Work

The consistent subset supports incremental numerical information under this
masked-current task, while broader gains remain dependent on the trained fit.
Current results cannot isolate optimization variation from patient composition
because both change across seeds. Last-epoch selection also leaves convergence
uncertain. No permission to reinterpret nonsignificance as no physiological
signal, or to lower the evaluation gate.

Fixed seeds71/83 remain pending. Multiplicity, fresh external evidence and
future trajectory/uncertainty validation remain unresolved. No runtime weights,
registry, interval, serving change or promotion. Patient rows remain private.
