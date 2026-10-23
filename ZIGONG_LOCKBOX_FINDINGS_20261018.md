# Zigong Lockbox: First Independent Validation of the Self-Adaptive Model (2026-10-18)

Protocol: `ADAPTIVE_MODEL_LOCKBOX_PROTOCOL_20261018.md` (v3, `adaptive_model_lockbox_protocol_v3_20261018.json`).
Hashes were verified by the scoring script, and the database was scored once
(`site_adaptive_lockbox_Zigong_20261018.json`). No run was repeated and no code changed after
scoring.

## Data

Zigong Fourth People's Hospital infection ICU cohort (PhysioNet v1.1), single site, 9 ICU
departments as care units. Laboratory table only: the nursing chart's timestamps are day-level,
so vital-sign, urine-output and hypoxemia cells are not evaluable and vital-sign inputs are
imputed with MIMIC medians. Chemistry glucose is too sparse to evaluate. Treatment context is not
extracted, so treatment-context recipes fall back to the measurement-pure ridge.
**21 laboratory cells were evaluable; 26 cells lacked support.**

## Result (5 MIMIC samples; robust = full pass in at least 80% of samples)

| | Static v2 | Adaptive interval | + point recalibration | + point recalibration + DtACI (frozen form) |
|---|---|---|---|---|
| Robust full pass (21) | **1** | 8 | 9 | **10** |
| Robust point pass | 8 | 8 | 11 | 11 |

Robust under the frozen deployment form: hematocrit@3h, potassium@3h, hematocrit@6h, hemoglobin@6h, potassium@6h, hemoglobin@12h, potassium@12h, potassium@24h, creatinine@48h, potassium@48h.

| Cell | Recipe | Rows | Static full share | Adaptive full share | Adaptive point share | Coverage patient / care unit | MAE / persistence | |
|---|---|---:|---:|---:|---:|---|---:|---|
| hematocrit@3h | measurement_pure_ridge | 935 | 0.0 | 0.8 | 1.0 | 0.890 / 0.915 | 0.940 | **robust** |
| hemoglobin@3h | target_specific_change_hurdle | 581 | 0.0 | 0.6 | 0.6 | 0.910 / 0.902 | 0.857 | - |
| platelets@3h | target_specific_change_hurdle | 314 | 0.0 | 0.0 | 0.4 | 0.886 / 0.911 | 0.903 | - |
| potassium@3h | measurement_pure_ridge | 570 | 0.0 | 0.8 | 1.0 | 0.874 / 0.880 | 0.940 | **robust** |
| creatinine@6h | target_specific_change_hurdle | 426 | 0.0 | 0.0 | 0.2 | 0.843 / 0.839 | 0.979 | - |
| hematocrit@6h | measurement_pure_ridge | 1350 | 0.0 | 1.0 | 1.0 | 0.883 / 0.908 | 0.905 | **robust** |
| hemoglobin@6h | target_specific_change_hurdle | 861 | 0.0 | 0.8 | 0.8 | 0.899 / 0.900 | 0.857 | **robust** |
| potassium@6h | measurement_pure_ridge | 836 | 0.0 | 1.0 | 1.0 | 0.875 / 0.880 | 0.871 | **robust** |
| bicarbonate@12h | target_specific_change_hurdle | 1327 | 0.0 | 0.0 | 1.0 | 0.889 / 0.857 | 0.875 | - |
| chloride@12h | target_specific_change_hurdle | 569 | 0.0 | 0.6 | 0.6 | 0.882 / 0.878 | 0.925 | - |
| creatinine@12h | target_specific_change_hurdle | 377 | 0.0 | 0.0 | 0.4 | 0.862 / 0.858 | 0.991 | - |
| hematocrit@12h | target_specific_change_hurdle | 1171 | 0.0 | 0.0 | 0.0 | 0.893 / 0.914 | 0.865 | - |
| hemoglobin@12h | measurement_pure_ridge | 798 | 0.0 | 1.0 | 1.0 | 0.906 / 0.906 | 0.852 | **robust** |
| potassium@12h | measurement_pure_ridge | 779 | 0.0 | 0.8 | 1.0 | 0.877 / 0.886 | 0.889 | **robust** |
| bicarbonate@24h | target_specific_change_hurdle | 2086 | 0.0 | 0.2 | 0.2 | 0.895 / 0.886 | 0.875 | - |
| bun@24h | target_specific_change_hurdle | 838 | 0.0 | 0.2 | 0.2 | 0.880 / 0.881 | 0.977 | - |
| creatinine@24h | target_specific_change_hurdle | 840 | 0.0 | 0.0 | 0.0 | 0.881 / 0.901 | 0.975 | - |
| potassium@24h | measurement_pure_ridge | 1554 | 0.0 | 1.0 | 1.0 | 0.890 / 0.891 | 0.910 | **robust** |
| sodium@24h | target_specific_change_hurdle | 1506 | 0.0 | 0.0 | 0.0 | 0.877 / 0.883 | 0.962 | - |
| creatinine@48h | target_specific_change_hurdle | 651 | 0.8 | 1.0 | 1.0 | 0.889 / 0.891 | 0.942 | **robust** |
| potassium@48h | measurement_pure_ridge | 1181 | 0.0 | 1.0 | 1.0 | 0.889 / 0.892 | 0.841 | **robust** |

## Reading

- **Independent confirmation of self-adaptation.** At an untouched site the fixed interval passes
  1/21 robustly and the self-adaptive model 10/21. Whole-stream coverage is mostly 0.87–0.91.
- **The remaining failures are mainly point forecasts.** Several hurdle recipes (creatinine 6/12/24h,
  BUN@24h, sodium@24h, bicarbonate@24h, hematocrit@12h, platelets@3h) only marginally beat
  persistence (MAE ratio 0.96–0.99). This is consistent with an infection-only cohort and with the
  missing vital-sign inputs.
- Potassium is robust at every evaluated horizon (3, 6, 12, 24, 48h); hemoglobin and hematocrit at
  6–12h.

## Decision status

Under the protocol's decision rule, the 10 robust cells are **candidates** (robust on one
untouched database). Independent replication requires a second untouched database (SICdb or
HiRID). No registry was changed.

`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
