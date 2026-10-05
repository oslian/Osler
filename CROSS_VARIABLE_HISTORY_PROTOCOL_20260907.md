# Cross-Variable History Protocol

Frozen before outcome inspection on 2026-09-07.

## Question

Under one cohort, one target-specific 3-hour label contract, and one set of
patient/hospital roles, does pre-anchor history from other physiological
variables add future information beyond the target's own history?

This audit precedes any end-to-end encoder retraining. It distinguishes a
representation failure from an absence of stable incremental signal.

## Fixed Data Roles

- eICU is the development source.
- `fit`, `validation`, `calibration`, `patient_test`, and `hospital_test` use the
  existing deterministic, patient-disjoint prospective-change split.
- NWICU is a previously opened external diagnostic set, not an untouched
  lockbox and not a source of model or threshold choices.
- Targets are `glucose@3h` and `map@3h`.
- Future labels, anchor freshness, and target-specific tolerance are exactly
  those implemented by `scripts.audit_prospective_change.valid_rows`.

## Fixed Inputs

All methods receive the target's own 24-hour measurement history summary.

1. `history`: target-own history only.
2. `current_cross`: target-own history plus the current non-target physiology,
   observed mask, and measurement age.
3. `cross_history`: `current_cross` plus measurement-pure non-target changes at
   1, 3, 6, 12, and 24 hours before the anchor. A change is valid only when both
   endpoints are observed, the historical row is causally available, and the
   underlying measurement timestamps differ. LOCF repetition is not a new
   observation.
4. `cross_history_placebo`: the same feature width and estimator as
   `cross_history`, but the entire non-target history block comes from a
   different patient. The real target-own history remains unchanged.
5. `release`: the frozen active release source.

Database, hospital, care-unit, future values, future treatment, labels, and
post-anchor measurements are excluded from every learned input.

## Fixed Model And Evaluation

- All learned methods use the same signed-delta risk and quantile head family,
  training options, patient-equal weights, probability calibration, and split
  conformal procedure already used by the representation-repair audit.
- One representative anchor per patient is used for evaluation.
- The primary incremental-information gate requires `cross_history` to beat
  `history`, `current_cross`, `cross_history_placebo`, and `release` by
  patient-level paired bootstrap, without material stable-state harm, on both
  eICU patient-heldout and eICU hospital-heldout roles.
- Risk and interval gates are reported separately and cannot rescue a failed
  point gate.
- NWICU is evaluated only after the source candidate is frozen.

## Decision Boundary

- Passing both eICU point gates justifies a later end-to-end observation-aware
  encoder experiment for that target; it does not promote a runtime forecast.
- Failure means raw cross-variable history has not shown stable incremental
  future information under this contract. It does not prove that no such
  information exists in higher-frequency, treatment-intent, or longer-term
  data.
- This experiment cannot modify release weights, registry, calibrators, or
  serving behavior.

