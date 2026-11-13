# Point-Only Correction Ablation

Fixed one-feature Huber residual calibration uses only the frozen model's
prediction, not variable values, missingness or measurement ages. Same source
calibration patients, leave-component-out queries, weights, target scaling,
epsilon1.35/alpha0.1/tol1e-5 as full Huber.300iterations with the previously
tested warm-start2700continuation only on convergence warnings. No parameter
search, new features, subgroup selection, or outer evaluation.

All47targets/fivefolds processed;163fully supported/converged.162also have
complete full-Huber reference results and form the matched comparison:

- Point-only MAE is lower than full numeric-context Huber in127/162.
- Point-only improves unchanged in54/162 (full Huber previously36/162).
- Point-only improves median offset in56/162.
- Point-only improves every reference in15/162.
- In97/162, BOTH point-only and full Huber fail to improve unchanged.
- Chloride and MAP improve unchanged mean MAE in allfivefolds; neither, nor
  any other target, beats EVERY comparator in allfivefolds.

These are equal-component source-CV mean comparisons, not multiplicity-adjusted
promotion tests. Overlapping folds are not independent replication. Source
input information, dimension and per-coefficient regularization burden differ
between arms;127/162supports harmful added correction complexity in this
setting, not a unique overfitting mechanism or proof extra physiology has no
value in a differently trained model.

Crucially, low-dimensional calibration also usually fails to improve the base.
Do not substitute the narrative "remove all cross-organ inputs and it is fixed."
The full shared encoder is unchanged and still supplies the original point.
This result rejects universal downstream correction, not shared body learning.

Two tests verify held-out truth cannot influence fitting and unsupported cases
remain explicitly unavailable. No runtime, encoder, interval, forecast or
promotion changed. Current masked reconstruction only.

Artifact: `artifacts/native_reconstruction_point_only_20260925.json`.
