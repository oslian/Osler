# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `11`; evaluable: `11`
- Point pass: `9`
- Interval pass, fixed → value-conditional: `2` → `5`
- Full pass, fixed → value-conditional: `2` → `4`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@12h | target_specific_change_hurdle | 7008 / 1448 | True | 0.796 → 0.823 | 0.829 → 0.846 | 0.900 / 0.917 | False → False |
| chloride@12h | target_specific_change_hurdle | 4707 / 1294 | True | 0.804 → 0.813 | 0.801 → 0.810 | 0.902 / 0.912 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15004 / 2965 | True | 0.759 → 0.855 | 0.766 → 0.873 | 0.900 / 0.930 | False → False |
| glucose@12h | observed_treatment_context_ridge | 15006 / 1167 | False | 0.809 → 0.867 | 0.771 → 0.844 | 0.902 / 0.918 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15018 / 1132 | True | 0.880 → 0.925 | 0.859 → 0.915 | 0.903 / 0.939 | False → True |
| hematocrit@12h | target_specific_change_hurdle | 6787 / 1362 | False | 0.836 → 0.879 | 0.844 → 0.881 | 0.900 / 0.926 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 7263 / 1387 | True | 0.880 → 0.894 | 0.870 → 0.881 | 0.900 / 0.923 | True → True |
| map@12h | measurement_pure_ridge | 15000 / 1143 | True | 0.855 → 0.878 | 0.850 → 0.872 | 0.905 / 0.915 | False → True |
| o2sat@12h | measurement_pure_ridge | 15010 / 964 | True | 0.738 → 0.752 | 0.754 → 0.771 | 0.906 / 0.920 | False → False |
| potassium@12h | measurement_pure_ridge | 7533 / 1445 | True | 0.873 → 0.883 | 0.905 → 0.911 | 0.904 / 0.902 | True → True |
| respiratory_rate@12h | measurement_pure_ridge | 15023 / 999 | True | 0.856 → 0.865 | 0.856 → 0.868 | 0.899 / 0.911 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
