# Prospective Change Readout: Fixed Two-Target Experiment

## Question

Freeze the released shared JEPA and its normalizer. For glucose@3h and MAP@3h,
test a signed-delta distribution readout, prospective change risk, and risk-binned
conformalized quantile intervals. This is a candidate experiment, not permission
to change release sources or reinterpret unsuccessful clinical gates.

## Fixed Methods

- Current release, persistence, last-observed linear trend (3h extrapolation).
- Direct readout: existing 311-dimensional target feature contract plus seven
  target-history features; HGB quantile heads at .05/.50/.95 and an unweighted-
  class probability classifier with patient-equal sample weights.
- Same-capacity shuffled-history control: recompute JEPA with a different patient's
  historical tokens, retain the patient's own current token/current state, and
  shuffle target-history summaries except current value and current age. Donors
  stay in the same fit/evaluation role and cannot be the same patient. This is an
  artificial control, not a deployable model or cross-patient information source.
- Same-estimator target-history-only control and simple linear trend. These test
  whether shared context adds value beyond this patient's recent target trajectory.
- Head capacity fixed: 80 iterations, 15 leaves, min leaf 40, L2 1, learning rate
  .05, no early stopping or tuning on external outcomes. One training seed.
- No expert-amplitude [0,1] constraint; median head directly predicts signed delta.
  Quantile crossing is sorted before calibration. Prediction is current + delta.
- Research change thresholds: glucose 20 mg/dL, MAP 10 mmHg, inherited unchanged
  from the previous diagnostic. They are not clinical diagnoses/alert thresholds.

## Data and Role Separation

- Existing eICU canonical event grid at 3h; strict NWICU 3h event examples for
  transfer evaluation. Only strict/exact future events after anchor, positive
  actual delta_t, and anchor age <=1h. MAP actual horizon tolerance +/-0.5h;
  glucose +/-1h. No imputed or nominal-only future label substitution.
- Hospital hash holds out about 20% of eICU hospitals. Any patient encountered at
  a held-out hospital is excluded from every development role.
- Remaining patient hash: 50% fit, 20% probability/decision validation, 10%
  interval calibration, 20% test. Each calibration/validation patient provides one
  hash-selected anchor. Fitting uses at most six hash-selected anchors/patient.
- Evaluate one representative anchor/patient for primary scores and a bounded
  chronological replay of every eligible anchor in those selected test patients.
  Never split a patient's encounters between roles. Never use future labels to
  select an anchor for being dynamic rather than stable.
- All transformations, probability calibration, risk boundaries, alarm thresholds,
  and interval radii are frozen in an artifact BEFORE reading NWICU outcomes.
- eICU fit overlap with original frozen encoder training is not re-audited. Head
  roles are disjoint, not proof that the original encoder never saw test patients.
  NWICU was previously inspected; transfer validation is not a new untouched lockbox.

## Probability and Interval Calibration

- Platt calibration on separate eICU validation patients; no class balancing that
  would silently change prevalence. Current release/linear trend risk scores are
  calibrated from absolute predicted delta; persistence risk is fit prevalence.
- Freeze each method's alarm cutoff to an empirical <=5% stable-case false-positive
  budget in source validation. Strict score > cutoff avoids tie inflation.
- Also report ROC recall at <=5% FPR on held-out labels, explicitly a ranking
  diagnostic, NOT a new external serving cutoff. Report actual FPR/recall at the
  fixed source cutoff and paired patient bootstrap on alarm outcomes.
- Define common low/high risk bins from the direct head's validation median risk.
  Bin membership is always predicted pre-anchor, never realized future change.
- Fit CQR expansion max(lower-y,y-upper,0) on disjoint calibration patients within
  each bin (minimum 100). Missing support means no interval for that bin, not a
  secretly substituted global interval. Keep release intervals unchanged as a
  comparator. Candidate/control/persistence experimental intervals use the same
  bin rule and calibration role. Marginal/conditional coverage under shift is
  empirical, not a theorem. Reference: Romano et al., NeurIPS 2019 CQR.

## Reporting and Gates

- Patient-equal MAE, paired differences vs release/persistence/controls, stable
  harm and changed-case signed/magnitude performance; Brier risk score.
- Per-patient 90% interval coverage, width, proper central interval score, and
  coverage in pre-anchor risk bins and retrospective stable/change strata.
- Seven overlapping patient evaluation resamples are diagnostics, NOT seven
  training seeds or independent replications. Hospital/known care-unit slices
  and late-in-stay replay are reported; late stay is not calendar-time validation.
- Directional alerts, deduplicated future-measurement events, alert recall and lead
  time to the next observed value in chronological replay. True onset may occur
  earlier; no claim of pre-onset detection or sensitivity in unmeasured periods.
- Passing screen requires improvement vs release and persistence plus same-capacity
  controls, no stable harm beyond 0.05*research threshold, coverage in [0.87,0.93]
  overall and predicted risk bins, and source FPR budget retained. Full promotion
  additionally requires original external/runtime gates and independent data.
- No tuning after reading failures, no weight/registry update, no clinical/causal
  or next-measurement-time model. Only aggregate reports and candidate weights.
