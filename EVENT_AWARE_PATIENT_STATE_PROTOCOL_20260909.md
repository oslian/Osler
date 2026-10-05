# Event-Aware Patient-State Protocol

## Question

For the already validated `glucose@6h` release source, can anchor-known new
measurements and observed treatment events determine whether a previously
resolved patient residual is still valid?

## Fixed design

- The released JEPA, target head, source selector and patient supports are
  frozen.
- eICU is development data. The pre-existing unseen-hospital role and MIMIC-IV
  are evaluation roles; MIMIC-IV is not described as an untouched lockbox.
- Predictions are prequential. A forecast error may update patient state only
  after its future measurement time has resolved. Repeated forecasts resolving
  to the same measurement are deduplicated.
- All measurement, freshness and treatment features are available at or before
  the anchor. Future treatment (`act_*`) is forbidden.
- The primary candidate is a gate over the simple resolved-error patient
  correction. The gate does not alter global JEPA weights.

## Arms

1. `frozen_history`: released forecast with no patient update.
2. `resolved_error_only`: always apply the causal running patient residual.
3. `event_aware`: apply that same residual only when a source-trained expected
   loss-difference model predicts benefit from anchor-known measurement changes,
   freshness and observed treatment history.

Controls are a state-only gate, a capacity-matched event-shuffled gate, and a
causal wrong-patient residual state.

## Selection and evaluation

The eICU patient development role is split by patient into fit and tune sets.
Estimator settings are fixed; only the apply threshold is selected on tune.
Evaluation reports patient-equal MAE, patient-level paired bootstrap confidence
intervals, direction accuracy after an observed treatment-context change, and
additional correction movement on physiologically stable no-event anchors.

Promotion requires the event-aware arm to significantly improve both the frozen
release and always-apply residual on unseen eICU hospitals and MIMIC-IV, and to
beat state-only, event-shuffled and wrong-patient controls. Otherwise it remains
`candidate_only` and runtime behavior is unchanged.

## Boundary

Treatment is factual observed context, not an intervention assignment or causal
effect. No row-level clinical data or patient identifiers may be committed.

