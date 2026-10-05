# Same-Map Regularization Intervention

One fixed intervention, not hyperparameter selection: multiply saved scaled
kernel LinearSVR C by10 (100/n ->1000/n). Keep ALL other parameters, scaler,
RBF bandwidth, landmarks, feature map, fitting labels/weights, median offset,
within-component shuffle and encoder unchanged. Clone the existing regressor;
relearn every coefficient and intercept, not just an output gain. Keep50,000
iteration cap; report nonconvergence without changing the budget on results.

All47 targets/all5 original source folds/four arms: relaxed_body/raw/process/
shuffle. Query rows/components remain disjoint from fitting, old predictions
must reproduce from saved code+weights. No outer rows, query tuning or clipping.
Source selection data have been reused; they are development, not a lockbox.

For each of body/raw, compare with native,body HGB,raw HGB,population,matched
relaxed process/shuffle,and corresponding original SCALED kernel. Family2*47*7.
Patient-fold-first MAE and component-equal MAE,>=30patients>=3components,
paired5000bootstrap95%upper<0, one-sided t p<.05/family and negative mean BOTH
levels. Allfour solvers/allfive folds converged required. No weakened thresholds.
Report all means, fitting improvement and family-corrected contrasts. Original
HGB remains the stronger reference; beating only a weak kernel is not success.

This intervention separates penalty-limited coefficient fitting from the fixed
feature map to a limited extent; one finite C does not characterize all possible
regressors. If no robust increment beyond strong readouts, stop this kernel
repair branch rather than expanding a query-driven sweep. No deployment,
weights replacement, calibration, future/clinical/causal claim. Private heads
and patient rows remain ignored, aggregate artifacts hash-bound.
