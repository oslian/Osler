# MIMIC-IV Unit-Safe Treatment Precision Findings

## Question

Can strictly pre-anchor, dose/rate-resolved treatment records improve factual
point forecasts enough to narrow calibrated 90% prediction intervals?

This is an observational forecasting audit. It does not estimate treatment
effects and does not permit causal or counterfactual claims.

## Extraction Repair

The previous treatment context mixed incompatible quantities such as `mg`,
`mL`, `mEq`, and `units` inside generic numeric sums. That representation was
deleted rather than retained behind a compatibility layer.

The replacement uses:

- canonical dose dimensions (volume, mass, mmol, mEq, drug units, energy);
- canonical rate dimensions, while preserving weight-normalized rates;
- route and source counts;
- time-weighted rates over the six-hour history;
- the rate active at the anchor;
- only the delivered pre-anchor exposure for an infusion crossing the anchor;
- eMAR dose, product amount, infusion rate, adjusted rate, and prior rate;
- target-specific numeric treatment fields plus general treatment presence.

The full rebuild processed `640,164` anchors and `11,722,374` relevant
administration evidence rows. It replaced the obsolete mixed-unit treatment
columns with `375` factual treatment-context columns. The final input continues
to exclude all `act_*` future-window fields.

## Point Forecast Gate

Baseline is the same factual forecast without treatment context. Candidate adds
strictly pre-anchor unit-safe treatment features. Placebo adds equal random
capacity. Each target must pass seven patient-heldout splits, care-unit
heldout, and chronological heldout against both baseline and placebo.

| Target | 7 patient splits | Care-unit | Time | Result |
|---|---:|---|---|---|
| glucose | 7/7 | pass | pass | validated |
| bicarbonate | 7/7 | pass | pass | validated |
| MAP | 7/7 point improvement | fail vs placebo | pass | fallback |
| potassium | 7/7 point improvement | fail | pass | fallback |

For glucose, care-unit MAE changed from `33.083` without treatment context to
`30.750` with unit-safe treatment context. The patient-equalized paired MAE
delta was `-1.966`; the candidate also beat the equal-capacity placebo.

For bicarbonate, care-unit MAE changed from `1.850` to `1.820`. The
patient-equalized paired MAE delta was `-0.033` and passed both comparators.

## Conformal Interval Result

Intervals use patient-equalized split conformal residuals and require heldout
coverage between `0.87` and `0.93`.

| Target | Patient 7-split width | Care-unit width | Time width | Coverage |
|---|---:|---:|---:|---|
| glucose | `127.73 -> 124.20` (`-2.37%`) | `126.91 -> 123.77` (`-2.47%`) | `126.14 -> 122.69` (`-2.74%`) | pass everywhere |
| bicarbonate | `7.53 -> 7.47` (`-0.74%`) | `7.50 -> 7.45` (`-0.67%`) | `7.47 -> 7.40` (`-0.89%`) | pass everywhere |

The intervals became narrower because the factual point forecast became more
accurate, not because the nominal coverage target was weakened.

## Boundary

Only glucose and bicarbonate may use the unit-safe treatment precision
increment. MAP and potassium retain their currently validated source or
persistence fallback. The artifacts contain aggregate audit results only;
patient-level data remain local. `causal_claim_allowed` remains `false`.

