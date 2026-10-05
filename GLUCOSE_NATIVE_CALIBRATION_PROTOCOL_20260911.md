# Fixed glucose output-scale diagnostic

Reuse native glucose predictions, labels and patient partitions; verify hashes
and original scores. No encoder changes. Before results, fix two readout tests:
patient-weighted median residual offset, and patient-weighted median regression
of actual delta on predicted delta with intercept (sklearn QuantileRegressor,
alpha=0, quantile=.5, solver=highs). Fit on source tune only, independently for
own and real arms. No coefficient bounds, search or evaluation selection.
Report negative slopes explicitly. No physiological or treatment-effect meaning
is assigned to the coefficients. The affine model may increase false movement.

Evaluate each against its own uncalibrated predictions, and corrected real
against identically corrected own on source evaluation and hospital patients.
Use existing patient-paired bootstrap, 2,000 draws. Keep both methods regardless
of outcome. Decompose original signed bias and delta correlation descriptively.
This is a diagnostic of global output scale, NOT conformal calibration, a
clinical decision rule, or a new untouched validation. Tune previously selected
checkpoints; held-out scores do not fit coefficients. No runtime changes.
