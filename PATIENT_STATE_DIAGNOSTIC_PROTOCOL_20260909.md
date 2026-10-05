# Patient-state failure localization protocol

Frozen before the diagnostic outcome report is generated.

## Question

Why did patient-local adaptation fail to improve the validated `glucose@6h`
release source across eICU hospitals and MIMIC-IV, and which component remains
worth developing?

## Fixed data roles

- The existing eICU patient roles are reused.
- eICU memory patients fit representation readouts.
- eICU patient-test patients select state hyperparameters.
- eICU retention patients calibrate source-domain intervals.
- eICU hospital-heldout and MIMIC-IV evaluate transfer.
- MIMIC-IV is repeated validation, not an untouched lockbox.

Every forecast requires observed anchor glucose and observed future glucose
5-7 hours later. A personalized row requires at least one earlier forecast
outcome to have resolved. Repeated forecasts resolving to the same future
measurement contribute only the most recent forecast error to patient state.

## Diagnostic 1: evaluation integrity

- Every placebo is built on the same full role frame and evaluated on the same
  personalized patient/anchor support as its candidate.
- A wrong-patient donor may only contribute a state from an equal or earlier
  relative stay time. No later donor state is legal.
- All seven placebo seeds use this identical contract.
- Point and interval gates must implement the comparisons stated here.

## Diagnostic 2: patient-state component ablation

Compare the frozen release with:

1. a shrunk patient residual level;
2. the same level with age decay but no slope;
3. age-decayed level plus regularized slope extrapolation.

Small eICU-only grids select prior strength and half-life. Hospital-heldout and
MIMIC are untouched by that selection. Each component is compared with the
previous simpler component, frozen release, persistence and seven causal
wrong-patient placebos. Stable and changed rows are retrospective diagnostics,
never routing inputs.

## Diagnostic 3: uncertainty-only personalization

The point prediction is frozen to the existing release for every method. Only
the uncertainty scale changes: constant population scale versus a patient
scale estimated from resolved prior release errors. Each receives its own
patient-disjoint conformal radius. Width is interpreted only when both methods
meet the same `[0.87, 0.93]` row and patient-equal coverage requirement.

Two questions are reported separately:

- zero-shot transfer of an eICU-calibrated radius to eICU hospital-heldout and
  MIMIC-IV;
- target-domain MIMIC calibration/test splits, explicitly diagnostic and not
  external validation.

## Diagnostic 4: information location

Equal-capacity target-specific regressors fit the same eICU memory patients and
predict signed glucose change from:

1. target history only;
2. target history plus other current observed values, masks and ages;
3. target history plus causal 1-24 hour cross-variable history;
4. target history plus the frozen JEPA latent;
5. target history plus a matched wrong-patient JEPA latent.

The purpose is localization, not promotion. Raw cross-variable gain without
latent gain points to encoder information loss. Neither gaining points to weak
incremental tabular supervision on this cell. Latent gain over both target
history and its placebo supports further encoder work but still requires a new
untouched lockbox before promotion.

## Boundary

This audit cannot modify runtime weights, source selection, calibrators or
serving behavior. It is factual, `candidate_only`, and
`causal_claim_allowed: false`.
