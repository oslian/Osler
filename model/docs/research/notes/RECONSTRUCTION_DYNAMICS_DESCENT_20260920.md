# Protected Body-Dynamics Descent Feasibility

Fixed bounded7. Different from prior body-update-gate probes: gradients are
through body_dynamics parameters only, using the exact masked reconstruction
SmoothL1. All33trained targets and both source groups, two disjoint source-fit
probes per task, up to64patients each; sparse tasks use floor(N/2). No actual
parameter update and no runtime change.

Desired descent is the normalized equal-group BUN gradient. Project it onto
the cone satisfying nonnegative dot product with every task gradient in the
source probe. SciPy bounded convex dual optimization; numerical constraint
residual must be >=-1e-6. A negative parameter step along this vector should
reduce the protected probe's losses only to first order. This is not Adam,
finite-step verification, or a physiological constraint.

## Results

| Source probe | Protected direction norm | Other-probe harmful dots, BUN-only | Other-probe harmful dots, protected |
|---|---:|---:|---:|
|0|.532893|31/66|27/66|
|1|.688419|28/66|20/66|

Both optimizations converge, minimum normalized constraint dots about-3.14e-8
and-5.92e-8. Both protected directions preserve positive BUN loss-descent dots
on both groups of the other probe (0: .00671/.01636;1: .00817/.02311).
Thus protection does not erase all BUN progress, but its non-harm property does
not transfer to the other patients' gradient estimates. Counts refer to66
group-target tasks, NOT distinct variables or measured accuracy regressions.
Threshold for counted negative raw dots was-1e-8; magnitude and finite-step
effects still matter. These are local descriptive results without uncertainty
or full promotion significance.

## Interpretation

A common local direction on one probe is feasible; impossibility is not proven.
But sample-dependent constraints fail to protect all tasks on a disjoint probe.
Do not introduce this projected gradient as a runtime repair or claim it safely
improves all targets. Existing small-gradient instability evidence is consistent
with this result, now localized to the implicated dynamics parameters.

An actual finite-step replay on separate source-selection patients is still
needed before attributing measured error changes to these directions. Training
guardrails need representative, independent support rather than assuming one
probe establishes non-harm. No new shared architecture/head justified here.

Three solver tests pass: conflict removal, zero feasible progress, zero gradient
constraints. Raw gradients remain private. All33trained/6pending scope retained.
Evidence: artifacts/native_reconstruction_dynamics_descent_20260920.json.
