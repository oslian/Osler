# As-of Activity-Conditional Intervals (v4): Findings (2026-10-15)

Script: `activity_conditional_interval_audit.py`. Detail tables:
`ACTIVITY_CONDITIONAL_INTERVAL_eICU_FINDINGS_20261015.md`,
`ACTIVITY_CONDITIONAL_INTERVAL_NWICU_FINDINGS_20261015.md`. Every v2 number reproduced the v2
audit exactly.

## Method

The eICU module "active" rules, restricted to measurement criteria and evaluated on as-of
values, are portable to MIMIC, eICU and NWICU and use no future information. Each target uses
its own module. Active and inactive rows get separate v2-style widths, calibrated on MIMIC only.
Half of the never-fitted MIMIC patients are used for calibration; the other half are a MIMIC
held-out check.

## Predeclared adoption rule: not met

| Rule | Result |
|---|---|
| 1. MIMIC held-out active and inactive coverage both in [0.89, 0.91] in ≥ 90% of cells | 25/46 (54%) — **not met** |
| 2a. eICU full passes not lower than v2 (non-independent) | 14 → 12 — **not met** |
| 2b. NWICU full passes not lower than v2 (non-independent) | 28 → 28 — met |

**v4 is not adopted. v2 remains the method.**

Rule 1 was strict for these sample sizes: the smaller group has about 787 patients per cell, a
binomial SE of about 0.011, so the ±0.01 band is about one SE. On the same held-out data v2 meets
it in 4/46 cells and v4 in 25/46. v4 does improve conditional calibration inside MIMIC. The rule
is applied as written; rule 2a fails independently.

## What v4 shows

| | Active v2 → v4 | Inactive v2 → v4 |
|---|---|---|
| MIMIC held-out | 0.890 → 0.899 | 0.911 → 0.898 |
| eICU | 0.848 → 0.860 | 0.872 → 0.863 |
| NWICU | 0.884 → 0.892 | 0.911 → 0.896 |

- **Inside MIMIC, as-of activity is a weak signal.** v2 already gives active rows 0.890; the
  value normalization captures most of it.
- **eICU active rows are much worse (0.848) than MIMIC active rows (0.890).** Whatever makes the
  eICU disease cohorts harder is not captured by as-of measurements. Diagnosis-based membership
  (v3) did capture a real effect inside MIMIC (members as low as 0.82), but it is neither portable
  nor available at prediction time.

## Conclusion of the conditioning series (v2 → v3 → v4)

| Version | Conditioning | Status |
|---|---|---|
| v1 | value bin, within-bin patient weights | removed (over-coverage) |
| v2 | value bin, normalized scale | **current method** |
| v3 | diagnosis-based disease module | not adopted (not portable; uses discharge codes) |
| v4 | as-of measured disease activity | not adopted (predeclared rule not met) |

MIMIC-only calibration cannot remove the remaining eICU disease-cohort shortfall with the
information available at prediction time. The practical remedy is the one Epic adopted:
**site- or cohort-local recalibration** with recent local labels, followed by monitoring.

`causal_claim_allowed: false`. Non-independent re-checks; nothing promoted.
