# External prequential patient-state validation

Frozen before outcome evaluation on 2026-09-09.

## Question

Can a patient-local, time-aware predict-update state improve the already
validated `glucose@6h` release source when each MIMIC-IV patient is replayed in
time order, while preserving 90% coverage with narrower intervals?

## Fixed data roles

- eICU patient-test rows select the time-aware state configuration.
- eICU retention rows calibrate method-specific normalized conformal scores.
- eICU hospital-heldout rows test source-domain transfer.
- MIMIC-IV is repeated external validation. It has already been inspected and
  is not an untouched lockbox.
- No MIMIC outcome may select state parameters, correction limits, interval
  normalization or conformal radius.

Every row requires an observed anchor glucose and an observed future glucose
5-7 hours later. The primary personalization slice requires at least one prior
forecast outcome to have actually resolved before the current anchor. Merely
being a patient's second row is insufficient.

## Compared predictions

1. `persistence`: the current observed glucose.
2. `frozen_release`: the current validated shared-JEPA target head.
3. `simple_patient_state`: the frozen release plus the shrunk mean of this
   patient's previously resolved release errors.
4. `time_aware_patient_state`: the frozen release plus a patient-local residual
   level and trend. Earlier errors decay by age; slope is regularized; the
   correction is bounded by the eICU residual scale.
5. `shuffled_patient_state`: the same time-aware correction and uncertainty
   scale taken from a different patient's time-matched state. Seven fixed donor
   permutations form the capacity-matched placebo.

The shared JEPA and release head remain frozen. No global model weight changes
during either database replay.

## eICU-only selection

The fixed grid is:

- residual half-life: 12, 24 or 48 hours;
- zero-centered prior strength: 1 or 3 observations;
- slope penalty: 4 or 16 in half-life-normalized time units;
- maximum retained resolved errors: 8;
- absolute patient correction limit: two eICU robust residual scales.

The lowest patient-equal MAE on the eICU patient-test personalized slice wins.
Ties prefer stronger shrinkage, then longer half-life, then larger slope
penalty. The grid is not changed after MIMIC results are visible.

## Prequential timing

- Each patient's anchors are processed in chronological order.
- A prediction is recorded before its future label is available.
- A residual enters patient state only when its actual future measurement time
  is no later than the current anchor.
- State is reset between patients and never transfers patient identifiers or
  residuals into another patient's prediction.
- Stable and changed strata use the future only for retrospective evaluation,
  never routing or state updates.

## Interval test

Each point method gets its own normalized split-conformal interval. The scale is
the method's pre-anchor estimate of residual uncertainty. One outcome-independent
row per eICU calibration patient determines the finite-sample 90% score
quantile. The frozen and persistence references use a constant eICU scale.

An interval is acceptable only when row and patient-equal coverage are both in
`[0.87, 0.93]` on eICU hospital-heldout and MIMIC-IV. Width is compared only
between methods that pass the same coverage requirement.

## Decision gate

The time-aware state may proceed toward release only if it:

- significantly beats the frozen release, simple patient state, persistence
  and every shuffled-history placebo on the personalized slice;
- passes in both eICU hospital-heldout and MIMIC-IV;
- does not trade stable-patient false movement for changed-patient gain;
- preserves 90% row and patient-equal interval coverage in both databases;
- has significantly smaller patient-equal interval width than the frozen and
  simple-state intervals.

This audit remains factual, `candidate_only`, and
`causal_claim_allowed: false`. It cannot modify runtime weights, registry,
calibrators or serving behavior.
