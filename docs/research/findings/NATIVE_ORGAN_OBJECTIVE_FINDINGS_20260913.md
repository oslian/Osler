# Native Replay Objective Ablation

Completed fixedteacher/centeredteacher/value-only eight-epoch paired pilot.
Source64fit32tune, same initialization and data. All selectepoch0; tune baseline
8.34714 versus final8.41928/8.42284/8.42367 respectively. Selected heldout
predictions exactly equal baseline, so no new efficacy or promotion.

Center is fit-only. Frozen teacher removes coordinate drift. Value-only removes
the auxiliary latent objective entirely yet still reproduces tuning failure.
Therefore the constant latent alignment shortcut is real as a loss diagnostic,
but is not necessary for the observed point forecast failure. Do not identify
it as the proven unique root cause. Centering alone is not a sufficient repair.
This pilot does not establish equivalence or full-data impossibility. All new
weights remain private/candidate-only; runtime and calibration unchanged.

Next diagnosis must address whether the residual target has a reproducible
relationship to available features on adequately supported source patients,
rather than treating another loss change as a presumed fix.
Evidence: `artifacts/native_organ_objective_pilot_20260913.json`.
