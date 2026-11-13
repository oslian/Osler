# Near-Miss Portfolio Protocol

Date: 2026-09-08

## Question

Do heterogeneous near-miss predictors contain complementary patient-level
forecast information that a development-frozen convex portfolio can recover?

This is not another unrestricted mixture-of-experts experiment. Previous work
already rejected free routing, additional decoder capacity, and same-family
seed ensembles as universal repairs. This experiment tests only the four cells
predeclared by the corrected whole-catalog source screen as having the closest
but domain-inconsistent source signals:

- `bun@24h`
- `map@3h`
- `bicarbonate@12h`
- `creatinine@6h`

## Frozen candidate family

For each cell, the candidate sources are:

1. persistence;
2. the active release prediction;
3. target-own causal history;
4. target-own history plus current non-target state;
5. target-own history plus current and 1/3/6/12/24-hour non-target history.

Exact duplicate columns are removed before fitting. The learned portfolio uses
non-negative weights constrained to sum to one. It therefore interpolates among
existing predictions and cannot create an unconstrained extrapolation.

A capacity-matched placebo portfolio keeps persistence, release and own-target
history unchanged, but replaces the current and temporal non-target information
with the matched wrong-patient views used by the source screen. Real and placebo
weights are independently fit by the same procedure.

## Patient roles

Hospital holdout assignment is outcome-blind and support-balanced, identical in
principle to the corrected catalog screen. All patients in the held-out hospital
fold are removed before the remaining patients are hash-split into:

- 50% source-model fitting;
- 20% portfolio weight tuning;
- 30% patient-heldout evaluation.

The hospital-heldout patients form the second evaluation role. Source models
are fit only on the fit role. Portfolio sources, duplicate removal, best-single
choice and seven-bootstrap simplex weights are frozen using only tune patients.

## Diagnostics and gate

The outcome-aware row oracle is reported only as an impossible upper bound. It
is never a model and never selects weights, patients or sources.

The frozen convex portfolio must beat, in both patient-heldout and
hospital-heldout roles:

- persistence;
- active release;
- tune-selected best single source;
- the equal-weight real portfolio; and
- the independently trained capacity-matched placebo portfolio.

Every paired patient-bootstrap 95% upper bound must be below zero, all seven
patient resample point estimates must be negative, and stable-case harm must
stay inside the existing target-specific allowance. Failure against any
reference keeps the result candidate-only.

`creatinine@6h` is additionally replayed on the locally available MIMIC-IV 6h
transition artifact using the eICU-frozen models and weights. That artifact uses
the older nominal-horizon label contract, so this is exploratory transfer
evidence, not a pristine lockbox or a substitute for the eICU strict-label gate.

## Boundaries

- No active checkpoint, source selector, interval calibrator or runtime registry
  may change.
- MIMIC-IV and eICU have both informed prior development; no promotion or new
  external-validation claim is allowed from this experiment.
- Reports contain aggregate metrics and frozen weights only, never patient rows
  or identifiers.
- This remains factual forecasting. `causal_claim_allowed` is `false`.
