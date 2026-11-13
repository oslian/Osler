# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `11`; evaluable: `11`
- Point pass: `8`
- Interval pass, fixed → value-conditional: `0` → `2`
- Full pass, fixed → value-conditional: `0` → `2`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@12h | target_specific_change_hurdle | 2945 / 1006 | False | 0.783 → 0.799 | 0.766 → 0.776 | 0.901 / 0.899 | False → False |
| chloride@12h | target_specific_change_hurdle | 1778 / 751 | True | 0.758 → 0.760 | 0.740 → 0.733 | 0.902 / 0.901 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15000 / 4813 | True | 0.749 → 0.788 | 0.723 → 0.767 | 0.902 / 0.899 | False → False |
| glucose@12h | observed_treatment_context_ridge | 1296 / 633 | False | 0.759 → 0.801 | 0.627 → 0.718 | 0.901 / 0.900 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15020 / 1124 | True | 0.870 → 0.872 | 0.843 → 0.851 | 0.906 / 0.903 | False → False |
| hematocrit@12h | target_specific_change_hurdle | 3029 / 937 | False | 0.809 → 0.802 | 0.818 → 0.814 | 0.899 / 0.899 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 3311 / 981 | True | 0.855 → 0.838 | 0.859 → 0.845 | 0.900 / 0.903 | False → False |
| map@12h | measurement_pure_ridge | 15004 / 1166 | True | 0.857 → 0.859 | 0.867 → 0.871 | 0.906 / 0.901 | False → False |
| o2sat@12h | measurement_pure_ridge | 15006 / 1080 | True | 0.837 → 0.841 | 0.839 → 0.846 | 0.904 / 0.899 | False → False |
| potassium@12h | measurement_pure_ridge | 3254 / 1038 | True | 0.862 → 0.870 | 0.870 → 0.897 | 0.903 / 0.899 | False → True |
| respiratory_rate@12h | measurement_pure_ridge | 15007 / 1051 | True | 0.867 → 0.877 | 0.872 → 0.885 | 0.899 / 0.904 | False → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
