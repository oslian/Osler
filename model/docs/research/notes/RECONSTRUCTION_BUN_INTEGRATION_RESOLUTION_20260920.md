# BUN Integration Resolution Diagnostic

Fixed bounded-site epoch7 weights, identical masked-current BUN pairs and all
original observations. Replay max_step_hours=1,0.5,0.25 on90selected/3086broad
source-selection patients. Original per-patient errors reproduce within1e-7.
This is not future forecasting, new external validation, or checkpoint fitting.

| Source group | 1h step MAE | 0.5h step MAE | 0.25h step MAE |
|---|---:|---:|---:|
| Selected90 |1.105465044|1.105248849|1.105099917|
| Broad3086 |0.893611644|0.893358542|0.893357129|

Normalized reconstruction MAE, not original BUN units. Broad quarter-step
minus original=-0.000254515, descriptive paired95%CI[-0.000345838,-0.000162065],
about0.0285%relative improvement. Selected quarter-step delta=-0.000365127,
CI[-0.000857963,0.000068441], crossing zero. Broad mean absolute prediction
shifts are0.000914810(1->0.5) and0.000527964(0.5->0.25).

## Interpretation

The tested step reductions change aggregate error far less than disabling body
dynamics (prior broad delta=-0.042459,4.751%). Coarse stepping is therefore not
a sufficient explanation of the observed broad BUN penalty in this checkpoint.
Do not claim convergence for all step sizes, patients, targets or checkpoints.
The learned field explicitly receives substep delta as input, so this changes
both numerical integration and learned field inputs; it is not a pure numerical
solver-error experiment. Similar aggregate MAE does not imply identical states.

Do not select a deployment step after inspecting this result. No runtime
checkpoint, registry, calibration or promotion changed. Global dynamics deletion
already harms most targets; remaining work concerns state-dependent learned
dynamics and the joint training objective, not a universal branch removal or
another finer-step sweep. This diagnostic does not establish a unique root cause.

Two context-manager tests pass (invalid step and restoration after exception).
Audit completed exit0 and restored original step after every forward pass.
Evidence: artifacts/native_reconstruction_bun_integration_resolution_20260920.json.
Patient-level predictions remain in ignored private data, not this report.
