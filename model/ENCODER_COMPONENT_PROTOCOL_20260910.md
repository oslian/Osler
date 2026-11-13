# Value And Age Sensitivity

Same seven32-epoch whole identity checkpoints as group removal, same source
support and exact cached full predictions. All11 variable groups plus global
non-target block. Always preserve glucose and its full history.

Two independent diagnostic interventions per group:
- Value: normalized observed values become checkpoint center0; mask, age and
  presence unchanged. Missing values stay0. This removes numeric content but
  retains availability pattern; it is NOT imputation or an observed value.
- Age: observed channels get constant3h (midpoint of the6h eligibility bound),
  missing age unchanged. All values, masks and presence remain unchanged.
  This removes exact age, not all temporal/missingness information. Repeated
  observations and masks across history can still encode measurement timing.

Full vs altered paired patient MAE,1000 bootstrap draws, all groups and seeds
reported. Constant interventions may be off-manifold; neither proves absence
of physiology or causal organ effects. Age also conveys physiological staleness,
not solely hospital behavior. No external selection or deployment, no claims
of independent lockbox or multiple-comparison-adjusted significance.
