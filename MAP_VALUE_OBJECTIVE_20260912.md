# MAP Value Objective Check

Re-evaluated selected own/full checkpoints' saved predictions using the exact
normalized SmoothL1 value objective (beta1, MAP scale16.3086mmHg) with
patient-equal aggregation. Parent bindings verified. Formula matches PyTorch
at zero, boundary, both signs and large residuals in a regression test.

Full value loss is numerically worse in5/7 fitting splits and all7 source-tune
splits. Thus the full-context failure is not confined to using MAE for reporting
instead of the actual value objective. This is a descriptive comparison, not
new significance or promotion evidence.

The combined training objective also includes JEPA latent loss. This check
does NOT measure latent loss, gradient competition, optimizer trajectories or
prove an L1 retraining intervention cannot help. Selected-epoch differences
remain. No model, runtime or calibration changed.

Evidence: `artifacts/map_value_objective_20260912.json`.
