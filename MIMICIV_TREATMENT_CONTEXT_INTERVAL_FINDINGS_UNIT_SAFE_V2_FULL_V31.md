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
- Validated targets: `bicarbonate, glucose`

| Target | Rows | Subjects | Treatment Features | 7-Split Selected | 7-Split vs Baseline | 7-Split vs Placebo | Careunit | Time | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| glucose | 80003 | 27283 | 149 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| bicarbonate | 76879 | 29302 | 139 | 7/7 | 7/7 | 7/7 | pass | pass | validated |

## Boundary

Only pre-anchor `hist_*` treatment context is a factual input. This audit does not allow causal, counterfactual, clinical, or treatment-recommendation claims.
