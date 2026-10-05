# Source-Site Constraint Metric Reconciliation

Re-evaluate the exact saved site-constrained heads on all source fitting
examples. No new fitting, head selection or threshold search. Verify parent
dependencies, coefficient hashes and complete feature digest. Recompute
original predictions in the same float64 coordinates as the constrained solve.
Numeric comparison tolerance is1e-7; this is not a statistical test.

## Results

All 692 target/group/site SmoothL1 constraints reproduce within tolerance.
However,94 of those combinations have increased MAE;45 of66 target/group pairs
contain at least one source hospital with increased MAE. Therefore the previous
'no source regression' statement must remain explicitly SmoothL1-specific.
These are repeated target/site combinations, not692 unique hospitals.

The metric mismatch is NOT sufficient to explain all transfer failures:
albumin/selected, AST/selected, bilirubin/selected, INR/broad and PTT/selected
regress on held patients despite no source hospital showing increased MAE.

PTT/selected is a particularly direct example: all8 source hospitals improve
MAE, with source patient-average change-.05074 and site-equal change-.05016.
Held-patient change is+.07566 and held-site change+.09878. This failure starts
at transfer, not merely at switching the scoring metric.

Glucose/broad illustrates the converse:7/11 source hospitals have higher MAE
despite lower SmoothL1; source overall MAE rises+.001914, yet held MAE improves
-.009502. Thus a source MAE mismatch also does not mechanically imply transfer
harm. BUN source MAE improves in every hospital of bothgroups; its remaining
incremental gate issue is not explained by this metric mismatch.

## Decision

Keep the training guarantee labelled accurately; do not call it universal
accuracy retention. A prospective MAE-aligned constraint experiment could test
whether that mismatch is repairable, but cannot be advertised as a complete
solution for PTT or the other counterexamples. Source no-harm constraints are
empirical, not future-hospital guarantees.

Dependency inspection: scipy is installed; cvxpy, OSQP and Clarabel are not.
If an exact MAE-constrained linear-head experiment is pursued, scipy HiGHS
linear programming with absolute-residual epigraphs is an available mature
solver. Changing L2 anchoring to an LP-compatible L1 penalty would change the
objective and therefore requires its own matched unconstrained control, not
comparison only against the previous SmoothL1/L2 arm. No such fit has run yet.

Two tests pass, including a concrete counterexample where SmoothL1 decreases
while MAE increases. Full audit exit0. No weights, runtime, intervals, promotion
or clinical/causal claims changed. Current masked reconstruction only.

Evidence: artifacts/native_reconstruction_site_constraint_metrics_20260920.json.
