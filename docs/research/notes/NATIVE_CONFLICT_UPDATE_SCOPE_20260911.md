# Actual trained parameter scope

Reconstructed initial native-expanded parameters and compared all7old/clean
full checkpoints. Parent hashes verified, optimizer parameter IDs unique.
Clean checkpoints change11belief-filter tensors,24variable-mixer tensors,
24module-mixer tensors,4state-encoder tensors,6shared-predictor tensors and
4value-head tensors in every split. Thus no frozen-context-only training
explanation is supported by these checkpoints.

Median block-relative update norms: history.03065,variablemixer.02158,
modulemixer.02284,stateencoder.04281,sharedpredictor.02500. Raw norms are not
comparable measures of causal contribution or useful information across blocks.
EMA target changes are expected despite exclusion from optimizer; declaration
as trainable alone does not imply a head received supervision.

No optimizer updates executed in this audit. This confirms training wiring,
not that the updates improve generalization. No runtime/calibration/promotion.
Artifact: `artifacts/native_conflict_update_scope_20260911.json`.
