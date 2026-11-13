# Value-Conditional Interval v2: Fixing v1 Over-Coverage (2026-10-15)

## Cause

v1 Mondrian intervals weighted patients equally *within each value bin*. A patient with a few
transient rows in an extreme bin, typically with large residuals, got full weight there, while the
patient-equal evaluation counts each row as 1 / (patient's total rows). Bin widths were inflated,
and MIMIC patient-fold coverage was 0.902–0.942 (median 0.914; only 11/47 cells in [0.89, 0.91]).

## Method selection (MIMIC only, predeclared rule)

`value_conditional_interval_method_dev.py` → `value_conditional_interval_method_dev_20261015.json`,
47 continuous cells, 5-fold patient CV inside MIMIC calibration rows.

| Method | Cells in [0.89, 0.91] | Coverage range | Median worst-bin error | Width vs fixed |
|---|---:|---|---:|---:|
| fixed | 47 | 0.897–0.907 | 0.074 | 1.00 |
| mondrian_v1 | 11 | 0.902–0.942 | 0.028 | 0.99 |
| mondrian_global_weight | 46 | 0.898–0.917 | 0.009 | 0.94 |
| **normalized_bin_scale (selected)** | **47** | 0.897–0.907 | 0.029 | 0.94 |

The rule selected `normalized_bin_scale`. `mondrian_global_weight` has better per-bin calibration
but missed the first criterion by one cell (`creatinine@12h`, 0.917). Choosing it now would be a
post-hoc change. `external_value_conditional_interval_audit.py` now implements v2 only; v1 was
removed.

## Re-check on eICU and NWICU (not independent)

Both databases were already used, so these results cannot support promotion. NWICU was run under
`nwicu_external_recheck_protocol_20261015.json` (`independent_validation: false`).

| | Fixed | v1 | v2 |
|---|---:|---:|---:|
| NWICU full pass (of 40) | 26 | 26 | **30** |
| NWICU median patient coverage | 0.897 | 0.914 | 0.901 |
| eICU full pass (of 49) | 12 | 21 | **16** |
| eICU median patient coverage | 0.855 | 0.880 | 0.860 |
| Cells with coverage > 0.93 (both databases) | – | 9 | **0** |

- **Over-coverage is gone.** No cell exceeds 0.93 on either database.
- **NWICU:** v2 passes 30/40. Relative to fixed it gains `respiratory_rate@6h`,
  `respiratory_rate@12h`, `glucose@24h`, `map@24h`, `creatinine@48h` and loses `hematocrit@12h`.
- **eICU:** v2 passes 16/49, fewer than v1's 21. The 7 cells v1 won and v2 lost (for example
  `bun@24h` 0.888 → 0.845, `creatinine@24h` 0.895 → 0.859) passed under v1 only because its
  inflated width happened to offset eICU's within-bin under-coverage (about 0.05, found in the
  2026-10-14 diagnostic). v1's eICU gain was partly an artifact of miscalibration.
- 15 cells pass both databases under v2. The set differs from v1: `hemoglobin@12h`, `map@12h`
  and `bun@24h` drop out; `heart_rate@24h`, `heart_rate@48h` and `map@24h` come in.

## What this means

- v2 is correctly calibrated wherever the population resembles MIMIC (MIMIC CV, NWICU).
- The eICU gap that remains is the within-bin shift: at the same current value, eICU coverage
  is lower. Better MIMIC-only calibration cannot remove it. It needs an explanatory covariate or
  site-local calibration (the Epic-style approach).

## Status

- The formal external result is still the NWICU v1 one-shot (26/40 eligible for review).
- v2 needs a new untouched database for independent validation. None is available on this
  machine (no HiRID, AmsterdamUMCdb or SICdb found; the MIMIC-III demo is the same hospital as
  MIMIC-IV).
- The runtime registry is unchanged.

`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
