# Whole-Catalog Cross-Variable History Screen Protocol

Date: 2026-09-08

## Question

Before retraining an observation-aware JEPA encoder, which continuous release
target-horizon cells contain stable incremental source information in raw
measurement-pure non-target observations?

This is a source-attribution screen, not a model promotion experiment.  Its
purpose is to avoid end-to-end retraining where the raw candidate information
already fails before the encoder.

## Frozen data and roles

- Use the canonical 5,000-encounter eICU whole-ICU event examples.
- Evaluate all 39 continuous cells in the 41-cell release catalog.  Binary
  event cells are outside this value-regression screen.
- Use the catalog horizon and target-specific exact/strict measured future
  label produced by the event ledger, including actual measurement time.
- Require a finite anchor target whose measurement age satisfies the release
  freshness contract.
- Split patients into fit and patient-heldout roles.  Assign hospitals to five
  deterministic folds and, separately for each target-horizon, select the fold
  whose eligible-patient support is closest to 20%.  This selection uses label
  availability only, never future values, errors or predictions.  Hold out its
  patients before fitting and never expose hospital identity to a model.
- Select rows without examining outcomes and use at most four fit anchors per
  patient.  Evaluation uses one deterministic anchor per patient.

## Fixed feature views

Every target uses the same estimator and fixed hyperparameters.

1. `own_history`: the target's causal 24-hour level, slope, variability,
   freshness, span, count and last delta.
2. `current_cross`: own history plus the other 52 variables' current normalized
   values, measured masks and ages.
3. `current_cross_placebo`: own history plus another patient's current block.
4. `cross_history`: current cross-state plus causal 1/3/6/12/24-hour non-target
   deltas and validity masks.  A carried-forward value is not a new observation.
5. `cross_history_placebo`: keep the real patient's own history and current
   cross-state, but replace only the temporal block with another patient's.

The placebo correction above is stricter than the first two-target audit: a
temporal candidate cannot win simply because its placebo lost the patient's
current physiology.

## Fixed screen

The model predicts signed future change.  A source component is eligible for
later encoder training only when:

- current-state semantics beat both target-own history and the equal-capacity
  current-state placebo, or temporal semantics beat both current cross-state
  and the equal-capacity temporal placebo;
- the paired patient bootstrap 95% upper bound is below zero in both the
  patient-heldout and hospital-heldout roles;
- all seven prespecified patient resamples have a negative point estimate; and
- stable cases do not suffer material movement relative to persistence.

Beating persistence is reported separately.  It is not required to establish
that an input contains incremental information, and source eligibility does
not authorize runtime promotion.

## Decisions

- Only eligible cells may proceed to end-to-end observation-aware encoder
  training.
- Failed cells are recorded and are not retrained under the same observation
  contract.
- No active checkpoint, target head, interval calibrator, selector or runtime
  registry may change.
- All outputs are aggregate factual research results.  No row-level patient
  data or identifiers may be written.
- `causal_claim_allowed` remains `false`.
