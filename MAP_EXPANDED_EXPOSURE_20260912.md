# Expanded MAP Exposure and Tune Curves

Exact replay of the unchanged training sampler; no model was fitted or selected
using this audit. All seven parent reports and their direct bindings verified.

- At epoch32, all fitting patients have been sampled, but only36.9-39.3% of
  eligible fitting rows have been sampled. Median19-20 draws per patient.
- At the selected full checkpoint, row coverage is20.6-35.4%; patient coverage
  is99.88-100%. This is not omission of a large fraction of patients.
- Best full tune MAE exceeds best own tune MAE in all7splits, by0.066-0.460mmHg.
- Full best last16epochs improves best first16 in3/7; own in5/7. The curves
  do not show a universally still-improving full model at the budget boundary.

Interpretation: expanding rows did not give exhaustive exposure; this limits
claims about convergence. It does not establish undertraining as the cause or
justify promising that more epochs will help. Training-set prediction errors
were not recorded, so these curves alone cannot distinguish underfitting from
in-sample overfit. The next discriminating measurement is selected-checkpoint
fit error versus unchanged tune error, with own and full on identical rows.

No promotion, runtime, interval or external-validation changes. Seven splits
overlap and are not independent replications. Row exposure is not unique
future-event exposure. Evidence: `artifacts/map_expanded_exposure_20260912.json`.
