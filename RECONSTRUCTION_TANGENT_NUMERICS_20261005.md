# E43 numerical amendment before held-out evaluation

The initial process34265 exited1 after computing all2048 fold0 fit Jacobians.
L-BFGS-B reached2000 iterations (gradient infinity norm7.761694346687535e-8),
and the required success flag was false. No step, model or role prediction
artifact had been written; fold0 directory was empty. No selection or same-source
performance was inspected. This is not evidence of original-model failure.

Original trainer SHA256:
`8a2d5ad60c66d81a56457b141762fea06f4d13e0f5d2111077b81df214891561`.
An unchanged snapshot is retained under the private experiment directory's
`failed_initial/fit_tangent_bridge.py` and bound in the completed report.
Original protocol SHA256:
`c34191a555decf13007d33c7e104872132433eb4376039e17352a9988bde1866`.
The original protocol is not rewritten and this amendment is disclosed.

## Numerical change, not a scientific setting change

Use the exact reversible coordinate transform q=z/sqrt(eigenvalue+lambda).
The same loss, lambda, initial zero, retained eigenbasis,2000 iteration limit,
solver tolerances and original-q gradient<=1e-5 acceptance check remain.
This rescales the solver's coordinates using the fixed fit Hessian upper bound;
it does not change the allowed corrections, optimum, patient roles, step search,
gate, targets or reference models. It changes the numerical trajectory and may
still fail on other folds. Failure remains an explicit stop, not acceptance.

Tests compare primal objectives, gradients and predictions and add a scaled
full-row-rank quadratic-region problem with an analytic ridge solution.
Only after these checks is the same five-fold experiment restarted. This is
an implementation amendment after a numerical failure, not an untouched
original implementation or a post-hoc successful-model selection.
