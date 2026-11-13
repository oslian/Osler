# Frozen nonlinear transfer diagnosis

Question: does the partial source-only raw nonlinear context gain survive
hospital and database transfer, without a JEPA encoder?

Refit only the 21 source-selected trees from potassium_tree_history_20260911.json.
Reuse original fit patients, weights, source-supported channels, preprocessing,
and selected leaf counts. No new hyperparameter search; tune/evaluation patients
never enter fitting. Require each source evaluation MAE to reproduce within
1e-10 before any external prediction. Save private models outside version control.

Use the existing potassium external protocol: glucose-available eligible anchors,
measured potassium current age <=6h and future 5-7h. This is not all potassium
patients. Compare full/own/support/persistence on identical rows; patient-equal
MAE and patient-paired 1000-bootstrap CI, seven source seeds. Report MIMIC care
units with >=30 patients without choosing a unit-specific model. No external
calibration, tuning or input selection. MIMIC is reused validation, not a pristine
lockbox. Split CIs are descriptive, overlapping and not multiplicity-adjusted.

Full vs own tests incremental context; full vs support checks numeric context
beyond observation patterns. A failure of both JEPA and trees is not proof that
information is absent. A tree gain alone does not identify a unique JEPA defect.
No runtime, uncertainty, causal claim, or promotion changes.
