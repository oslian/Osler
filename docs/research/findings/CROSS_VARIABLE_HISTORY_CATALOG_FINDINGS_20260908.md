# Whole-Catalog Cross-Variable History Findings

Date: 2026-09-08

## Decision

Raw non-target current state and causal 1-24 hour history did not identify a
release cell that justifies end-to-end observation-aware JEPA retraining under
the present tabular contract.

Of the 39 continuous cells in the 41-cell catalog:

- 37 had at least 100 patient-heldout and support-balanced hospital-heldout
  patients and were evaluated;
- `platelets@1h` and `platelets@3h` had only 77 and 89 hospital-heldout patients
  and remain insufficient-support results;
- 0 passed the current cross-state eligibility gate;
- 0 passed the temporal cross-history eligibility gate.

This is a source-attribution result, not evidence that the active release is
ineffective.  Target-own history models frequently beat persistence.  The
failed claim is narrower: after target-own history is known, the other 52
variables do not provide stable incremental information across both new
patients and held-out hospitals under this feature contract.

## Corrected placebo and hospital design

The first run used the earlier fixed hospital fold and a 200-row threshold.
Only 12 cells were evaluable, so that run is retained only as a superseded
pilot in `artifacts/cross_variable_history_catalog_20260908/report.json`.

The canonical run fixed two evaluation defects before inspecting its results:

1. Hospital folds are deterministic and the fold closest to 20% of eligible
   patients is selected separately per target-horizon using support only, never
   outcome values or model errors.
2. The temporal placebo preserves the real patient's target history and current
   cross-variable state; only the lagged non-target block is replaced by a
   time-matched wrong patient's block.  A separate current-state placebo tests
   the current non-target block.

The corrected aggregate report is
`artifacts/cross_variable_history_catalog_20260908_v2/report.json`.

## Closest signals

No near signal passed both domains and the stable-patient guard.

- `bun@24h` current cross-state improved patient-heldout MAE over own history by
  `0.1906` (`95% CI [-0.3012, -0.0769]`) and over its matched placebo by
  `0.1717`.  In held-out hospitals, the corresponding own-history comparison
  was `-0.1505` with CI `[-0.2883, +0.0067]`; it crossed zero.  Stable-case harm
  also exceeded the predeclared allowance in the hospital role.
- `map@3h` temporal history beat current cross-state and temporal placebo on
  patient-heldout data by `0.1099` and `0.0890` MAE respectively.  Both hospital
  intervals crossed zero, and stable cases were significantly harmed.
- `bicarbonate@12h` temporal history passed both paired patient comparisons,
  but neither hospital comparison and neither stable-case guard passed.
- `creatinine@6h` temporal history passed both hospital comparisons, but was
  indistinguishable from its references on patient-heldout data.

These opposite-domain and stable/dynamic reversals are exactly what the screen
was intended to catch.  They cannot be repaired by relaxing the gate or by
placing the same features behind a larger encoder.

## What this resolves from the unfinished list

### Observation semantics in an end-to-end encoder

The prerequisite source screen failed for every sufficiently supported release
cell.  End-to-end retraining is therefore not authorized for this tabular
contract.  A future retry must introduce a genuinely different information
contract, such as beat-level pre-event signals or richer treatment intent, not
the same current values, masks, ages and 1-24 hour histories behind a larger
network.

### Cohort and anchor alignment

The whole-ICU eICU event-example cohort now exists, but the old AKI source is
AKI-selected and onset-anchored while the available MIMIC transition artifact
is admission-relative and lacks the same event-ledger future-label metadata.
A fair one-factor anchor experiment still requires rebuilding the MIMIC
canonical event ledger/examples or rebuilding the AKI subset from the same
canonical ledger.  Comparing the two current files would change population,
anchor and label policy simultaneously and is not a valid isolation test.

### Verifiable organ stores and fluxes

The current eICU ledger contains measured urine output but not a complete,
unit-normalized intake and net-balance stream.  It cannot supervise an exact
body-fluid store.  Existing organ exchange remains predictive message passing,
not material flux.  Exact conservation must wait for a complete intake/output
ledger; algebraic laboratory formulas remain audits only.

### Multi-timepoint joint supervision

The canonical eICU examples already contain 1/3/6/12/24/48-hour measured future
targets.  Persistent recurrent training across this grid was previously run
and rejected at the point gate (`0/38` in the formal pilot), and corrected
target-specific residual experiments did not activate.  This catalog screen
also covered all release horizons.  Repeating that architecture without a new
source signal would duplicate a completed negative experiment.

### Untouched external validation

MIMIC-IV and NWICU have already informed development.  No HiRID or
AmsterdamUMCdb data is present locally, so no newly tuned method can receive a
pristine lockbox claim in this run.

## Runtime boundary

- No checkpoint, target head, interval calibrator, source selector or runtime
  registry changed.
- The active 41-cell release remains authoritative.
- The screen is aggregate, factual, research-only evidence.
- `causal_claim_allowed` remains `false`.
