# MIMIC-IV Observed Treatment Context Accuracy Findings

This audit asks whether observed treatment context improves factual forecasts. It does not estimate treatment effects.

## Gate

- Baseline: ridge forecast without `hist_*` / `act_*` action fields.
- Candidate: same forecast plus observed pre-anchor `hist_*` treatment features.
- Placebo: baseline plus an equal number of random-noise features.
- Pass rule: candidate must beat both baseline and placebo across 7 patient-heldout splits, careunit-heldout, and time-heldout.

## Result

- Cohort rows: `640164`
- Subjects: `63307`
- Validated targets: `glucose`

| Target | Rows | Subjects | Treatment Features | 7-Split Selected | 7-Split vs Baseline | 7-Split vs Placebo | Careunit | Time | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| glucose | 80003 | 27283 | 65 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| potassium | 80008 | 25316 | 89 | 7/7 | 7/7 | 7/7 | fail | pass | fallback |
| map | 80022 | 8125 | 99 | 7/7 | 7/7 | 7/7 | fail | pass | fallback |
| bicarbonate | 76879 | 29302 | 58 | 7/7 | 6/7 | 7/7 | fail | pass | fallback |

## Boundary

Only pre-anchor `hist_*` treatment context is a factual input. This audit does not allow causal, counterfactual, clinical, or treatment-recommendation claims.
