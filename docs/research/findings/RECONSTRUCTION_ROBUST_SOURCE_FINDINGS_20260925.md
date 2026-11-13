# Robust Source Correction: Partial Mean Gains, No Stable Replacement

All47targets x5folds processed using ONLY existing calibration-role patients
and leave-one-component-out refits. No new outer prediction file was opened,
no new outer evaluation, no hyperparameter sweep. Same snapshot features,
support rule, frozen encoder and component weights as the ridge contrast.
Huber epsilon1.35/alpha0.1/max_iter300/tol1e-5 were fixed before execution.

## Failure Concentration Check

From the prior163complete source-CV comparisons,138had mean numeric-ridge harm.
127of these had harm in a majority of components;122remain harmful after
removing ANY single component from the diagnostic mean. No component was
actually excluded. The failure is not generally explained by one bad site.

## Robust Objective Results

162target-folds have fully supported AND converged robust refits. Within
exactly these162:
- Numeric Huber improves unchanged mean MAE in36; numeric ridge in25.
- Huber improves ridge in83.
- Huber improves unchanged, median offset, both ridge corrections and
  process-context Huber simultaneously in13.
- AST alone improves unchanged across allfivefolds, but it does NOT beat all
  references in allfive. No target wins every reference in every fold.

Thirteen allreference mean wins: albumin0; alkaline_phos0/1; ALT0/2/4;
AST0/2/3/4; creatinine0; PTT3/4. These are source mean comparisons, not corrected
significance tests, runtime cells or independent replication. Fold fits share
patients/components across folds, so counts must not be treated as independent
Bernoulli trials. No target chosen for deployment from these counts.

Five component-refits have at least one nonconverged arm: amylase fold1 (one)
and TSH fold0 (four). They are explicit nonconvergence, not successful zero
corrections; incomplete groups cannot support success. Missing role/domain
support also remains explicit. All numerical predictions computed were finite.

## What This Establishes

This robust-loss alternative repairs some mean-error failures but is not a
general transferable correction. It does not establish that extreme residuals
are the sole cause; Huber estimates residual scale and its regularization
semantics also differ from ridge. The prior35fit-level objective mismatches
and111fit-good/source-held-component-bad examples remain separate phenomena.

No global robust replacement and no outer rerun is justified by this source
evidence. The evidence points back to domain-dependent conditional relations,
sampling uncertainty and feature/state limitations; it does not uniquely
identify a biological cause. Further repairs need source component-level
evidence, not another outer-driven head search or relaxed gate.

Two new tests verify query truth cannot alter fitting and unsupported fitting
is explicitly unavailable; five prior correction tests also ran. Dependency
hashes and convergence status recorded. No weights, runtime, interval, registry,
forecast or promotion changed. Current masked reconstruction only.

Artifact: `artifacts/native_reconstruction_robust_source_20260925.json`.
