# Bounded Amplitude Calibration: Frozen Protocol

## Question

Can state-dependent adoption of a fixed expert's proposed change reduce
overshoot without discarding useful dynamic predictions? This is a candidate
readout on the same frozen shared JEPA, not a new physiology encoder.

Five targets at 6h: creatinine, BUN, urine output, potassium, bicarbonate.
Reuse the exact cohorts, expert checkpoint and patient partitions from
`expected_error_routing_20260905_v2` and `forecast_aware_routing_20260905`.
Do not tune on MIMIC, change label windows, or relax the existing gates.
MIMIC is repeatedly inspected transfer validation, not an untouched lockbox.

## Fixed Candidate and Controls

Prediction: `anchor + alpha(state, expert evidence) * frozen_expert_delta`.
Alpha is constrained to [0,1]. It is an adoption coefficient, NOT a clinical
event probability. A zero expert proposal remains exactly zero. This readout
cannot repair a wrong direction or an underprediction requiring alpha > 1.

Use existing sklearn HistGradientBoostingRegressor: absolute-error loss,
80 iterations, learning rate .05, 15 leaves, minimum leaf 40, L2=1,
early stopping disabled. Fit `clip(actual_delta/expert_delta,0,1)` weighted
by `abs(expert_delta)`; this is equivalent to minimizing forecast MAE within
the bounded adoption family. Only disjoint eICU router-fit patients supply
these training labels. Exclude zero proposals from coefficient fitting.

Primary is fixed in advance: conditional adoption on the previous 227
shared-state, pre-anchor context, and forecast-evidence features. No model
selection or hyperparameter search in this audit.

Controls:
- Same-capacity model with all conditioning features jointly shuffled across
  patients, while recipient expert proposals and outcomes remain unchanged.
- Global MAE-optimal coefficient, fitted by weighted median on the same rows.
- State-blind, forecast-evidence-only coefficient (8 inputs).
- Shared-state plus evidence without care-context (168 inputs).
- Previous eICU-selected forecast-aware binary selector, unchanged.
- Always persistence and always fixed expert.

The shuffled model is a capacity control, the global model tests whether mere
uniform shrinkage explains the gain, and evidence-only tests the contribution
of patient state beyond the proposed forecast itself.

## Uncertainty

Never multiply uncertainty width by alpha. Recenter each fixed expert sample
distribution at that arm's point delta, preserving its unscaled spread.
Persistence uses its existing empirical delta distribution, centered at zero.
Fit one split-conformal scale per arm on disjoint eICU calibration patients.
Report 90% coverage and CRPS of the resulting *scaled* sample distribution.
All arms use this same calibration procedure; these distribution scores are
not directly comparable to the previous audit's unscaled CRPS.

## Evaluation and Boundaries

Reproduce previous fixed-expert, persistence and binary-selector point MAE.
Report patient-paired bootstrap, seven patient resampling seeds, care-unit
balanced and late-relative-time splits, source hospital holdout, and MIMIC.
These seven resamples are not seven independently trained checkpoints.
The late split is not a calendar-time lockbox; unit-balanced inference does
not prove success in every unit. Also report per-unit coverage descriptively.

Primary must pass point comparisons against all listed controls (including
global/evidence-only), existing coverage band .87-.93, and CRPS versus the
recalibrated empirical persistence distribution in every evaluation role.
Report stable/dynamic error using future truth ONLY for retrospective audit,
never as an input or routing label at inference.

Keep runtime unchanged even if this bounded audit succeeds, pending full
release gates and a new independent lockbox. All results remain factual,
candidate-only; no patient rows or identifiers in findings or reports.
