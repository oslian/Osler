# E44: weak parameter-sensitivity directions, fixed-weight ablation

## Question

E43 fits a full-rank local sample Jacobian but transfers poorly, even for the
linearized diagnostic. Is that failure concentrated in the fitted correction's
weakly supported parameter directions, or does the retained correction also fail?

This is distinct from the old 64-dimensional state-readout bottom16 ablation
and the landmark RBF fit-objective audit: the operator is the actual recurrent
prediction Jacobian with respect to E43's single 64x448 shared bridge. No new
training, regularization search, target selection or alternate body state.

## Fixed Analysis

- Require completed and independently verified E43 and unchanged parent hashes.
- Recompute exactly the same zero-bridge fit Jacobian J, one fold at a time.
  Keep original float64 event times. Check zero predictions and saved Jw.
- Eigendecompose JJ^T in float64 with the E43 numerical-rank threshold.
  Define strong eigenvectors by eigenvalue >= E43's original lambda; weak by
  retained eigenvalue < lambda. No rank sweep or threshold chosen from outcomes.
- Orthogonally project the SAVED float32 direction onto these right-singular
  subspaces. Report omitted numerical-null/roundoff residual separately.
  Check reconstruction, pairwise orthogonality and energy conservation.
- Fix the original E43 accepted alpha. On every original fit, selection and
  same-source query save the strong-only and weak-only linear predictions and
  actual nonlinear predictions after casting each direction to float32.
  Also replay the unchanged full E43 model and require exact stored parity.
- Linear contributions must reconstruct the original saved tangent slope up to
  1e-7*(1+abs(original_slope)); quantify cast/null residual, never conceal it.
  Do not add nonlinear ablation outputs together: recurrent updates are nonlinear.
- Summarize all53 targets, including unsupported, in patient-equalized and
  component-equalized MAE/SmoothL1. Compare each ablation to zero/full bridge.
  Report mean reversals and counts descriptively; no promotion or new gate pass.
- Independently recompute scalar losses and aggregates from saved outputs;
  use direct_prediction to check actual recurrent ablations. Bind inputs,
  code, protocol, private outputs and aggregate report. No raw rows tracked.

## Interpretation And Limits

Small singular value does not imply physiological noise; the spectrum depends
on parameterization, task scales and fitting support. Large weak-direction norm
alone cannot diagnose transfer failure. A useful test is whether removing those
directions improves actual query truth error while preserving the rest exactly.

If the strong-only correction still loses to the zero bridge, weak-direction
amplification is not a sufficient explanation. If only the linear diagnostic
improves, the finite-step gap remains. Any useful actual ablation is still a
development result requiring a separately frozen validation, not deployment.

Same existing five folds/20,480 queries, repeated patients aggregated as before;
fold0/2 not independent. MIMIC/NWICU are reused development, not new lockboxes.
The raw reader's greater fitting support remains a limitation. Current masked
reconstruction does not demonstrate future forecasting or self-learning.

Use existing SciPy eigh and PyTorch autograd/functional_call. Keep Jacobian in
RAM, one fold at a time; maintain10GiB free disk. Stop on binding, identity,
decomposition or parity failure. Do not change old trainer or checkpoints.
