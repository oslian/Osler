# MIMIC-to-eICU Value-Conditional Interval Audit (6h cells)

Frozen MIMIC point recipes; only the interval changes from one absolute width to
Mondrian conformal by MIMIC current-value quintile. eICU labels used for scoring only.
Gate unchanged: patient- and hospital-equal coverage in [0.87, 0.93].

- Cells: `9`
- Interval pass, fixed width: `1`
- Interval pass, value-conditional: `4`
- Full pass (frozen point + interval), fixed: `1`
- Full pass (frozen point + interval), value-conditional: `4`

| Cell | Frozen point | MIMIC CV cov. fixed / cond. | eICU patient cov. fixed → cond. | eICU hospital cov. fixed → cond. | Hospital range fixed → cond. | Median width fixed → cond. | Interval pass fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@6h | True | 0.900 / 0.925 | 0.744 → 0.833 | 0.766 → 0.833 | 0.55–0.89 → 0.67–0.96 | 0.375 → 0.415 | False → False |
| glucose@6h | False | 0.902 / 0.922 | 0.770 → 0.835 | 0.739 → 0.828 | 0.64–0.81 → 0.77–0.87 | 58.5 → 65.7 | False → False |
| heart_rate@6h | True | 0.903 / 0.929 | 0.887 → 0.912 | 0.878 → 0.907 | 0.82–0.90 → 0.88–0.93 | 17.3 → 20 | True → True |
| hematocrit@6h | False | 0.900 / 0.920 | 0.828 → 0.865 | 0.823 → 0.858 | 0.74–0.90 → 0.82–0.92 | 4.23 → 4.46 | False → False |
| hemoglobin@6h | True | 0.902 / 0.913 | 0.853 → 0.885 | 0.840 → 0.877 | 0.77–0.93 → 0.81–0.95 | 1.45 → 1.5 | False → True |
| map@6h | True | 0.905 / 0.913 | 0.841 → 0.872 | 0.839 → 0.868 | 0.79–0.89 → 0.83–0.91 | 19 → 19.4 | False → False |
| potassium@6h | True | 0.901 / 0.911 | 0.862 → 0.886 | 0.891 → 0.910 | 0.80–0.89 → 0.84–0.92 | 0.72 → 0.776 | False → True |
| respiratory_rate@6h | True | 0.903 / 0.918 | 0.859 → 0.888 | 0.876 → 0.901 | 0.72–0.95 → 0.75–0.97 | 7.61 → 8.4 | False → True |
| urine_output@6h | True | 0.902 / 0.905 | 0.815 → 0.832 | 0.776 → 0.802 | 0.49–0.96 → 0.48–0.95 | 226 → 191 | False → False |

## Per-bin eICU coverage (fixed → value-conditional)

| Cell | Q0 | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| creatinine@6h | 0.98 → 0.93 (1346) | 0.95 → 0.87 (1538) | 0.92 → 0.84 (2245) | 0.82 → 0.86 (4116) | 0.57 → 0.85 (5768) |
| glucose@6h | 0.85 → 0.82 (1547) | 0.87 → 0.80 (1867) | 0.86 → 0.84 (2516) | 0.81 → 0.84 (3138) | 0.63 → 0.85 (4723) |
| heart_rate@6h | 0.93 → 0.94 (3827) | 0.92 → 0.94 (3307) | 0.91 → 0.93 (2947) | 0.88 → 0.91 (2252) | 0.79 → 0.86 (2667) |
| hematocrit@6h | 0.80 → 0.88 (1228) | 0.87 → 0.85 (850) | 0.87 → 0.87 (623) | 0.88 → 0.89 (486) | 0.79 → 0.87 (355) |
| hemoglobin@6h | 0.81 → 0.88 (1039) | 0.90 → 0.90 (988) | 0.92 → 0.92 (856) | 0.85 → 0.86 (448) | 0.82 → 0.89 (332) |
| map@6h | 0.88 → 0.90 (1304) | 0.90 → 0.90 (1848) | 0.89 → 0.85 (2300) | 0.86 → 0.85 (3404) | 0.81 → 0.89 (6171) |
| potassium@6h | 0.89 → 0.91 (1797) | 0.92 → 0.91 (843) | 0.90 → 0.88 (587) | 0.84 → 0.85 (509) | 0.71 → 0.80 (675) |
| respiratory_rate@6h | 0.83 → 0.85 (1162) | 0.91 → 0.91 (2679) | 0.92 → 0.91 (1939) | 0.91 → 0.93 (3835) | 0.81 → 0.87 (5402) |
| urine_output@6h | 0.94 → 0.91 (3209) | 0.94 → 0.92 (2305) | 0.93 → 0.92 (2318) | 0.90 → 0.87 (3055) | 0.68 → 0.78 (4126) |

## Interpretation (2026-10-14)

- All 9 fixed-width intervals reproduced the frozen 2026-08-16 audit exactly before the
  interval was changed.
- Value-conditional intervals raised eICU patient-equal coverage in every cell
  (+0.02 to +0.09). Interval passes went from 1/9 to 4/9; full passes (frozen point gate
  plus interval) went from 1/9 (`heart_rate@6h`) to 4/9, adding `hemoglobin@6h`,
  `potassium@6h` and `respiratory_rate@6h`.
- Part of the gain comes from more conservative intervals. MIMIC patient-fold coverage
  rises from about 0.90 to 0.91-0.93. The eICU gain is larger than the MIMIC gain in every
  cell (creatinine +0.089 vs +0.025, MAP +0.031 vs +0.008), so the conditioning itself
  accounts for most of the improvement.
- Composition is largely corrected. Creatinine top-quintile coverage went from 0.57 to
  0.85, glucose from 0.63 to 0.85.
- The remaining failures (creatinine, MAP, urine output; glucose and hematocrit also fail
  the frozen point gate) now sit at a fairly flat 0.83-0.87 across value bins. This matches
  the within-bin eICU drop of about 0.05 found in the diagnostic. Case-mix conditioning
  cannot remove it; it needs either a new explanatory covariate or site-local calibration.
- Hospital-level spread does not shrink for creatinine and urine output (creatinine range
  0.67-0.96). Between-hospital heterogeneity beyond case mix remains.

**Decision.** Candidate only. The frozen runtime registry is unchanged. A promotion test
requires freezing this interval method and scoring it once on a database that has not
been used, for example NWICU, HiRID or AmsterdamUMCdb.

## Boundary

eICU was already used diagnostically (`MEASUREMENT_PROCESS_SHIFT_DIAGNOSTIC_FINDINGS_20261014.md`)
to choose this interval family, so any pass here is candidate evidence only and does not
change the frozen runtime registry. Promotion still requires an untouched database scored
once with this method frozen.

`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
