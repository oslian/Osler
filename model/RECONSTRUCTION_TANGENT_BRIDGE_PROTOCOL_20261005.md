# E43: full-fit local bridge solve and nonlinear transfer

Fixed before evaluating the new method. Current masked reconstruction, not
future forecasting, self-learning, clinical efficacy or release promotion.

## Question and competing explanations

E42 excludes hard dependence of the supported affine output rows, but not
reachability through the actual recurrent bridge. E36/E37 differentiated along
the already trained direction. E38/E40 modified stochastic updates. None solves
the fitting-set residual in the full bridge-parameter tangent space.

Keep the same frozen epoch10 model, bound organ proposals, 64x448 zero-start
bridge and all five existing folds. Compute the prediction Jacobian J with
respect to that single bridge. Do not introduce another state or output head.
Each fold uses the existing 2048 fit patients/contexts; selection and same-source
each retain their existing 1024 queries. Fold0/2 are not independent replicates.

## Fixed intervention

1. At bridge zero compute f0 and J using PyTorch autograd. Times remain float64.
   Preserve identical input/truth/role contracts and check zero predictions
   exactly against saved bound_start. No imputed targets or new patients.
2. Form K=J J^T in float64. Retain positive eigenvalues above
   max(n,p)*float64-eps*max_eigenvalue; report the numerical retained dimension,
   not an information-theoretic or exact recurrent rank. Reject materially
   negative eigenvalues. No Jacobian or patient rows in tracked artifacts.
3. Solve sum_i SmoothL1(f0_i+(J w)_i-y_i,beta=1)+lambda/2*||w||^2,
   lambda=0.001*trace(K)/n, using existing SciPy L-BFGS-B in the row-space basis.
   Start at zero; maxiter=2000, maxls=50, ftol=1e-12, gtol=1e-8. Require success,
   no objective increase, and reduced-space infinity gradient <=1e-5.
   Zero J returns exact zero correction; nonfinite inputs fail closed.
4. Cast the resulting w to the actual float32 bridge. Recompute Jw from the
   stored direction, not an unattainable float64 surrogate. Use original fit
   truth only for Armijo backtracking (c=1e-4), steps 1,1/2,...,1/128.
   Accept the FIRST step satisfying the regularized true recurrent fit
   objective; otherwise retain exact zero. Do not look at selection to pick
   regularization, step or checkpoint. Only one linearization, no iterative
   relinearization or hyperparameter search in this experiment.
5. Freeze the candidate. Replay actual nonlinear outputs and local-linear
   predictions at the accepted step on all three roles. Save separate
   f0, Jw, linearized prediction and actual prediction; the tangent model is
   diagnostic only and cannot replace the recurrent model at runtime.

## Verification and interpretation

- Unit tests compare the solver against a direct parameter-space SciPy solve,
  including conflicting identical inputs, null-space and nonfinite cases.
- For the first example of every represented target in each role, compare
  the float32 Jacobian projection to a float64 central finite difference at
  h=0.001 and 0.0005 along the normalized learned direction. Tolerance
  2e-5*(1+abs(slope)); no patient-specific choice of direction or steps.
- Reload saved candidate weights and independently replay with the existing
  direct_prediction implementation; tolerance 1e-7 absolute. Verify all
  non-bridge tensors unchanged and hash-bind old and new inputs/artifacts.
- Recompute MAE and SmoothL1 with scalar math and patient/component aggregation.
  Record optimizer convergence, fit line-search trials, nonlinear-linear gaps,
  train/selection/same-source reversals and unsupported targets explicitly.
- Preserve all E40 controls plus actual_step: eleven comparisons, patient and
  source-component paired statistics, >=30 patients and >=3 components,
  ci95 upper <0, p_less <0.05/(53*11), 5000 bootstrap repetitions. Do not drop
  difficult targets or report multiple folds as independent patients.
- Linear fit gain alone excludes neither memorization nor shift. Linear gain
  that disappears in the actual bridge indicates a finite-step/approximation
  limitation in this intervention. Real fit gain with held-out reversal
  means better fitting is not a sufficient transfer repair. Full screen gain
  still requires later forecast, calibration and new-lockbox work.

## Costs and stop conditions

This changes optimization budget and regularization as well as the update rule;
it is not a pure causal comparison of optimizers or a claim that Adam is buggy.
The bounded fit support remains much smaller than the raw reader's fit support,
as already diagnosed in E24/E25. Gram truncation and regularization limit what
a negative tangent result can rule out. Backtracking may choose zero.

Use one fold at a time and keep at least10GiB disk free. Parent caches must
already exist; never silently rebuild them. Completed bound artifacts are
immutable. Nonfinite values, changed identities, failed solver or replay stop
interpretation. No modifying the protocol after observing partial results.
No runtime, release weights, gates, intervals, commits or uploads changed.
