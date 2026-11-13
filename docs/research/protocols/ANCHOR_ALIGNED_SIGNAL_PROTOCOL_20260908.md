# Anchor-Aligned Cross-Database Signal Protocol

## Question

For `creatinine@6h`, does non-target physiology add reproducible information
after the patient's own creatinine history, and does the frozen shared JEPA
retain that information?

This is a source-attribution experiment, not a runtime promotion procedure.

## Frozen data contract

- Development database: eICU.
- External database: MIMIC-IV.
- Both databases are rebuilt with the same canonical event-ledger code.
- Both use 5,000 outcome-independent ICU encounters.
- Both use a 6-hour lookback, 3-hour anchor step, 96-hour maximum episode, and
  the same target-specific future-label policies.
- The target is `creatinine@6h` only.
- A row is eligible only when anchor creatinine is fresh under the release age
  policy and the future value is a real, strict post-anchor measurement.
- Treatments are excluded from both ledgers so the comparison isolates
  measurement history and shared representation.

eICU defines fitting, patient-heldout, and hospital-heldout roles. MIMIC-IV is
external exploratory validation; it has one hospital, so care-unit strata are
reported instead of a hospital split. Neither database is an untouched
lockbox.

## Frozen model arms

All learned regressors use the same HistGradientBoostingRegressor family,
patient-equal fitting weights, update budget, and source-only fitting data.

1. `own_history`: current creatinine, slope, variability, age, observed span,
   observation count, and last distinct change.
2. `raw_cross_history`: own history plus measurement-pure current and 1/3/6/
   12/24-hour histories of every non-target checkpoint variable, with explicit
   masks and ages.
3. `shared_jepa`: own history plus the frozen release's shared context,
   patient belief, target future latent, owner-organ latent, base value, and
   normalized measurement state.

Capacity-matched controls replace the added raw or JEPA block with a
time-matched block from another patient while preserving own history. No
future value, future measurement time, hospital identifier, care unit, or
database identifier enters a model feature.

## Evaluation

- One deterministic representative anchor per patient.
- Physical-unit MAE.
- Patient-level paired bootstrap confidence interval for candidate minus
  `own_history` and candidate minus its wrong-patient placebo.
- Seven patient resamples.
- eICU patient-heldout and hospital-heldout evaluation.
- MIMIC overall and supported care-unit evaluation.

## Predeclared interpretation

- Raw cross-history passes externally, JEPA passes externally: cross-variable
  signal transfers and the encoder retains it.
- Raw passes externally, JEPA fails: the encoder loses usable information.
- Raw fails externally, JEPA fails: pre-anchor cross-variable signal itself is
  not stable across databases under the aligned contract.
- Raw fails externally, JEPA passes: the nonlinear shared representation
  recovers a transferable signal absent from the fixed raw readout.

No result changes the 41-cell serving registry. `causal_claim_allowed` remains
`false`.
