# Near-Miss State Routing Protocol

Date: 2026-09-08

## Rationale

The predeclared fixed convex portfolio failed all four near-miss cells, while
the outcome-aware source oracle passed in both patient-heldout and
hospital-heldout roles. This establishes an error-complementarity upper bound,
but not a deployable gain. The next narrow question is whether legal pre-anchor
state can identify which source will have lower expected error.

## Frozen design

Use the identical four cells, eICU patient roles, fitted source models and
candidate sources from `NEAR_MISS_PORTFOLIO_PROTOCOL_20260908.md`.

For each tune patient, construct router inputs from:

- target-own causal history;
- every candidate source's signed proposal relative to the current value;
- absolute proposal sizes;
- source disagreement mean, standard deviation and range.

The router never receives a future value, realized change label, hospital ID,
care-unit label or future treatment. One maintained
`HistGradientBoostingRegressor` per source estimates that source's conditional
absolute error. At inference the source with the lowest non-negative estimated
error is selected. Persistence is a full source, so abstention requires no
separate threshold.

The capacity-matched placebo repeats the entire fit with current-cross and
cross-history predictions produced from matched wrong-patient inputs. It keeps
persistence, active release and own-target history unchanged.

## Gate

The frozen state router must beat persistence, active release, the tune-selected
best single source, the fixed convex portfolio and the independently fitted
placebo router in both patient-heldout and hospital-heldout roles. Existing
paired patient bootstrap, seven-resample and stable-case harm rules apply.

Source-selection frequency and outcome-oracle agreement are diagnostics only.
No subgroup, threshold, feature or model is changed after evaluation.

## Boundaries

This is a source-domain research diagnostic informed by the preceding oracle
result, not a promotion or untouched external validation. The active release is
immutable. Reports are aggregate-only. `causal_claim_allowed` remains `false`.
