# Forecast-aware selector repair: frozen protocol

Question: the existing selector receives 160 shared/context-belief features and
59 care-context features, but not the actual expert delta, quantiles or explicit
anchor target. Does exposing its decision alternatives improve conditional selection?

Reuse exactly the five 6h experts, source patient roles, and release bindings from
`artifacts/expected_error_routing_20260905_v2`. Do not refit the experts. Before
fitting each selector, assert patient-disjoint expert-fit, selector-fit, decision
selection, interval-calibration, patient-test and hospital-test partitions.

Add eight features: current observed target, target measurement age, expert delta,
expert delta q05/q50/q95, absolute proposed delta and expert quantile width. Divide
value/delta features by the existing fixed target dynamic threshold; age stays in
hours. No future label, actual future measurement time, hospital identity or treatment
after the anchor is an input. Expert quantiles are not claimed calibrated intervals.

Two candidate objectives with the same HGB budget (80 iterations, 15 leaves, min leaf
40, learning rate .05, L2 1): direct mean absolute-error benefit; cost-sensitive binary
choice weighted by the absolute realized error difference. Cost score is not a
calibrated event probability. Both train only on selector-fit patients not seen by
the fixed experts. This follows held-out stacking; existing sklearn estimators are
sufficient, with no added dependency. Reference:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html

Controls: old primary, old q25, always-persistence, always-expert, original-feature
mean/cost routing, capacity-matched shuffled added evidence, and no-care-context
variants. All added feature columns are shuffled jointly across patients for the
placebo; original shared state and care context remain unchanged.

Each candidate chooses a margin using eICU decision-selection MAE, then selects the
primary between the two real-evidence objectives on the same development partition.
Mean margins are (0,.02,.05,.1,.2)*target threshold; cost-score margins are
(0,.1,.2,.4,.6). No external selection. Separate eICU patients calibrate intervals.
All decisions are serialized before opening MIMIC outcomes for evaluation.

Same gates: paired patient MAE bootstrap, seven patient evaluation resamples,
care-unit-balanced and late-stay comparisons, CRPS versus empirical persistence,
90% coverage in [.87,.93]. Require superiority to old primary, persistence, expert,
same-objective blind/shuffled/no-care-context controls. Also report stable/dynamic
errors and wrong-direction versus overshoot contributions, not just aggregate MAE.
No thresholds are relaxed and no MIMIC recalibration is allowed.

This is a candidate diagnostic, not promotion: seven resamples are not seven refits;
late-stay is not calendar-time-heldout; small care units may have noisy estimates;
prior encoder training exposure is unaudited; MIMIC is already inspected. A new
lockbox and comparison to the current validated runtime router remain necessary.
Do not claim a richer selector proves a complete physiological state or causality.
