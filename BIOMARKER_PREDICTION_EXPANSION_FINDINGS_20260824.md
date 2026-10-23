# Biomarker Prediction Expansion Findings (2026-08-24)

## Decision

The expanded biomarker contract is implemented, but no new biomarker forecast
or abnormal-risk target is promoted. Next-measurement-time prediction is
explicitly excluded from this phase.

## Implemented task boundary

- Routine numeric candidates: lipids, HbA1c, liver function, CBC, iron
  metabolism and thyroid tests.
- Sparse numeric-or-risk candidates: CRP, vitamin D/B12, folate, ferritin,
  coagulation tests, cortisol and sex hormones.
- Risk-only candidates: troponin, tumor markers and rheumatoid factor.
- Observed-only: germline, hereditary cancer, pharmacogenetic and molecular
  residual-disease tests.
- Every task remains factual, candidate-only and non-causal.
- `next_measurement_time_modeled` is fixed to `false`.

## MIMIC-IV v3.1 support audit

Seven newly registered targets had enough repeated 0.5-72 hour measurements
to justify a numeric audit:

- HbA1c
- total, HDL and LDL cholesterol
- iron
- TIBC
- transferrin

Six had enough reference-range and class support to justify an abnormal-risk
audit: all of the above except LDL cholesterol.

Vitamin D, vitamin B12, folate and sex hormones did not have enough repeated
future observations. Tumor and autoimmune markers did not have enough future
class support. They remain direct observations or unsupported candidates; the
system must not manufacture a future number or risk.

## Numeric irregular-time forecast result

- Targets: 7
- Time bins per target: 0.5-6h, 6-24h and 24-72h
- Evaluated cells: 21
- Fully validated cells: 0

No raw-value or log-delta model passed all seven patient-heldout splits,
care-unit heldout and within-stay forward-trajectory heldout. Each cell stays
on persistence or missing according to its existing factual contract.

## Future abnormal-risk result

- Targets: 6
- Fully validated targets: 0

The candidate had to beat current abnormal-status persistence, discovery-set
prevalence and a capacity-matched shuffled-history placebo. No target passed
all seven patient splits plus care-unit heldout.

The failure modes are informative:

- HbA1c abnormal status was perfectly persistent in the care-unit split
  (`persistence_brier = 0`), leaving no legitimate short-term gain.
- Total cholesterol, HDL, iron, TIBC and transferrin sometimes improved over
  one baseline, but not over all baselines and placebo across patient splits.
- Repeated-pair support was only 104-386 pairs and was close to one pair per
  subject, so there was little patient history from which to learn dynamics.

## Runtime policy

- These tests expand the canonical observation schema, not the promoted
  forecast registry.
- Existing validated liver, CBC, thyroid, coagulation and troponin behavior is
  unchanged.
- New lipid, HbA1c and iron targets may be shown when directly observed.
- They may only be nowcast or forecast after an independent audit passes the
  existing patient, domain, placebo, uncertainty and external validation
  gates.
- Genetics remains observed-only or testing-needs triage; ordinary blood
  values are not substitutes for genetic assays.

## Artifacts

- `MIMICIV_BIOMARKER_PREDICTION_SUPPORT_20260824.json`
- `MIMICIV_BIOMARKER_IRREGULAR_FORECAST_20260824.json`
- `MIMICIV_BIOMARKER_ABNORMAL_RISK_20260824.json`

All reports are aggregate-only and contain no row-level patient data.
