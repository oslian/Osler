# Frozen Additional-Domain Outer Evaluation

All ten epoch8 checkpoints must be complete and hash-verified before inference.
Fold2reuse is explicit and not an independent replication. Evaluate every
accepted patient-target exactly once using that patient's TEST fold only.
No choice of models, epochs, inputs or normalization uses these outcomes.
No additional training or calibration occurs in this evaluation.

Compare real and process-only models and the same fold's fit-patient center
on all47targets. Persist raw truth/predictions and the fit scale. Report raw
MAE per target and fit-scale normalized MAE for cross-target aggregation;
never average unscaled lab/vital errors into a whole-body score.
Patient-targets are unique. Report patient-equal and connected-hospital-
component-equal errors, full per-component results, support and missing cases.

After all folds: paired patient and component mean differences versus both
controls,95% bootstrap intervals, and descriptive one-sided t-tests. Any
descriptive screen requires at least30patients and3components, negative means
at both aggregation levels, and all four p-values below0.05/47 (intersection
of both controls and both aggregation levels). Unlike the old39-target helper,
the denominator is the full47-target family. Screen status is NOT promotion.
Report all results, including thin domains and harmed components. Overlapping
training folds/reused functions make these conditional descriptive statistics,
not independent training-replica inference or new external validation.

No persistence baseline is appropriate for the fully masked current target:
its own history and siblings are unavailable by design. This does not replace
future forecasting gates. All final model improvements remain candidates
pending the existing external/calibration requirements. No runtime changes.
