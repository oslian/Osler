# Nonlinear raw-history diagnostic

Same inputs, patient weights and three-way fit/tune/evaluation masks as raw
Ridge comparison. Use installed sklearn HistGradientBoostingRegressor:
absolute_error loss,200iterations,learning_rate .05,min_samples_leaf30,
l2_regularization10,max_bins63,early_stopping=False,random_state=split seed.
Select max_leaf_nodes from7,15,31 using tune patient-MAE only. Never use sklearn
random row-wise internal validation (disabled). Own/full/support all same grid.

Seven seeds x three feature arms x three capacities=63fits. Primary incremental
full vs own; also full vs support, persistence and frozen full JEPA on identical
evaluation patients. No new external data or model selection on evaluation.
This is a diagnostic benchmark, not replacing shared-body architecture. A
negative result narrows this nonlinear family only, not all nonlinear relations.
Patient-weighted loss does not make tree min-leaf count patient-based. Multiple
comparisons descriptive; no causal, calibration or promotion claim.
