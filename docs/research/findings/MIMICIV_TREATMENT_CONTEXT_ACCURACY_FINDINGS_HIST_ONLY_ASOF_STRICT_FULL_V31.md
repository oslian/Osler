# MIMIC-IV Observed Treatment Context Accuracy Findings

This audit asks whether observed treatment context improves factual forecasts. It does not estimate treatment effects.

## Timing Contract

- Every forecasting arm excludes `act_*` fields from the future prediction
  window.
- Only pre-anchor `hist_*` features are candidates. They include observed
  administration flags, counts, completed amount-like doses, and active rates.
- For an inputevent still open at the anchor, its final total amount is
  excluded because it contains future delivery; only its already-known rate is
  retained. eMAR point administrations use their charted dose at start time.

## Gate

- Baseline: ridge forecast without `hist_*` / `act_*` action fields.
- Candidate: same forecast plus observed pre-anchor `hist_*` treatment features.
- Placebo: baseline plus an equal number of random-noise features.
- Pass rule: candidate must beat both baseline and placebo across 7 patient-heldout splits, careunit-heldout, and time-heldout.

## Result

- Cohort rows: `640164`
- Subjects: `63307`
- Validated targets: `bicarbonate, glucose, map`

| Target | Rows | Subjects | Treatment Features | 7-Split Selected | 7-Split vs Baseline | 7-Split vs Placebo | Careunit | Time | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| glucose | 80003 | 27283 | 160 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| potassium | 80008 | 25316 | 160 | 7/7 | 7/7 | 7/7 | fail | pass | fallback |
| map | 80022 | 8125 | 160 | 7/7 | 7/7 | 7/7 | pass | pass | validated |
| bicarbonate | 76879 | 29302 | 160 | 7/7 | 7/7 | 7/7 | pass | pass | validated |

## Boundary

Only pre-anchor `hist_*` treatment context is a factual input. This audit does not allow causal, counterfactual, clinical, or treatment-recommendation claims.
