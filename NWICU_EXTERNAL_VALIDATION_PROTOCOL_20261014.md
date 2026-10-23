# NWICU External Validation Protocol (frozen 2026-10-14T18:33:05-07:00)

Frozen before any NWICU label was read. Machine-readable version with code hashes:
`nwicu_external_validation_protocol_20261014.json`. The scoring script refuses to run if any
hashed file has changed.

## Question

Do the frozen MIMIC-IV recipes, with **value-conditional (Mondrian by current value) 90% intervals
calibrated on MIMIC only**, pass the external gates on an untouched database?

## Data

- Training and calibration: MIMIC-IV v3.1 only, transitions rebuilt with the current extractor
  (`*_treatment_unit_safe_v2.parquet`, 6h anchors, 6h look-back).
- External: NWICU v0.1.0, all ICU stays, same anchor/as-of/label code path
  (`nwicu_observation_transition_extract.py`). No prior project use of NWICU was found.
- At most 15,000 rows per database per cell, sampled by patient with a fixed seed.

## Not evaluable by design

- `urine_output` at 1h/3h/6h: NWICU has no output events.
- The 6 `observed_treatment_context_ridge` cells: NWICU has no input events.

## Declared deviations

- MAP derived as (SBP + 2*DBP)/3 from same-time arterial or non-invasive pairs; MIMIC MAP is device-measured mean
- target_specific_change_hurdle recipes read pre-anchor treatment features; these are absent in NWICU and imputed with MIMIC training medians
- variables absent in NWICU (ventilator settings, blood gases other than pH, biomarker expansion) are imputed with MIMIC training medians

## Methods and gates

- Primary interval: value_conditional: Mondrian patient-cluster conformal, 5 current-value quintile bins with edges from MIMIC calibration rows only, nominal 0.90, bins with <30 calibration patients use the pooled width.
- Secondary interval (comparison only): fixed: one patient-cluster conformal half-width (reported for comparison, not used for decisions).
- Point gate: patient-cluster paired bootstrap 95% CI upper < 0 versus persistence and versus the recipe's matched comparator (hospital axis not applicable: single hospital).
- Interval gate: patient-equal and care-unit-equal coverage both in [0.87, 0.93] (care-unit replaces hospital: single hospital).
- Binary event cells: point gate only (Brier versus persistence and MIMIC prevalence).
- Full pass: point gate and primary interval gate.

## Decision rule

A full pass makes a cell eligible for promotion review as an externally replicated candidate. No automatic runtime promotion. Failures stay on their existing validated expert or persistence.

## One-shot rule

The NWICU scoring command is run once. If it crashes, only the crash may be fixed; any fix must leave methods, gates and cohorts unchanged, and is reported with the frozen hash mismatch it causes.

`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
