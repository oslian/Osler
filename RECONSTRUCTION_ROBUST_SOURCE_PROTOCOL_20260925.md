# Source-Only Robust Objective Follow-up

Previous correction source-LOCO harms138/163;127have majority component harm,
122remain harmful after removing any one component (diagnostic only, no actual
component exclusions).35/189full fits improve MSE while worsening MAE.

Test one fixed robust-loss alternative BEFORE any new outer evaluation:
sklearn HuberRegressor epsilon1.35, alpha0.1, max_iter300, tol1e-5.
Same frozen numeric base, residual/fit-scale target, weighted scaler and
component-equal weights(sum1), per-variable support and features as the fixed
ridge correction. Process-context and numeric-context both fitted. No new
features, subgrouping, clipping, parameter search or activation selection.
Huber is a scale-estimating robust objective, not exact MAE or identical ridge
penalty semantics. Effects cannot be attributed solely to outlier removal.

For all47targets/fivefolds, use correction calibration roles ONLY and leave
one entire component out. Refit all input support/scaling. Compare unchanged,
median offset and prior identical-fold ridge with both robust arms. Record
nonconvergence separately; failed/nonfinite fits are not successful zero
corrections. Incomplete comparisons cannot establish gains.

Report source equal-component MAE, harmed components, and comparisons versus
allreferences. No outer prediction file is opened by this follow-up. It is
source development evidence, not promotion or an untouched confirmation.
No runtime, intervals, future prediction or encoder weights are changed.
