# Frozen cross-input and scale isolation

Seven old/clean full checkpoints crossed with both raw input histories.
Each uses its checkpoint-bound channel mask and event transform. All14 diagonal
predictions reproduce the training artifact bit-exactly. No fitting/selection.
Allowed channels and event centers match7/7; event scales differ6/7.

At fixed old weights, clean-input minus old-input hospital median paired MAE
effect is+0.001817, significant gain/harm0/0. At fixed clean weights, effect
is+0.002824, gain/harm0/0. Source gain1/7 for each, harm0/7.

A second isolation uses clean weights, old inputs, and old preprocessing.
Replacing clean event scale with old scale changes hospital error by median
0.000000243, gain/harm0/0. With input AND preprocessing fixed to old, swapping
to clean weights yields hospital median+0.290297, gain0/harm2; source+0.172811,
gain0/harm1. Thus this repair's observed pooled regression follows the learned/
selected checkpoint, not inference-time scale or direct replay differences.

This localizes this experiment, not every historical generalization failure.
Input changes during training can alter optimization and selected epoch; those
two effects remain inseparable from final checkpoints alone. Fixed-weight
crossing is diagnostic, not a valid new promoted model or a reason to undo
correct input semantics. Original28fits are the matched training evidence.
No runtime, calibration, promotion, or untouched external validation.

Artifacts: `artifacts/native_conflict_cross_20260911.json`,
`artifacts/native_conflict_scale_20260911.json`.
