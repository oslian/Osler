# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `11`; evaluable: `11`
- Point pass: `8`
- Interval pass, fixed → value-conditional: `0` → `1`
- Full pass, fixed → value-conditional: `0` → `1`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@12h | target_specific_change_hurdle | 2945 / 1006 | False | 0.785 → 0.796 | 0.769 → 0.775 | 0.902 / 0.897 | False → False |
| chloride@12h | target_specific_change_hurdle | 1778 / 751 | True | 0.765 → 0.758 | 0.740 → 0.734 | 0.902 / 0.902 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15000 / 4813 | True | 0.747 → 0.788 | 0.716 → 0.766 | 0.900 / 0.900 | False → False |
| glucose@12h | observed_treatment_context_ridge | 1296 / 633 | False | 0.766 → 0.819 | 0.666 → 0.736 | 0.901 / 0.902 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15020 / 1124 | True | 0.871 → 0.877 | 0.844 → 0.857 | 0.905 / 0.902 | False → False |
| hematocrit@12h | target_specific_change_hurdle | 3029 / 937 | False | 0.806 → 0.795 | 0.818 → 0.812 | 0.901 / 0.897 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 3311 / 981 | True | 0.852 → 0.840 | 0.857 → 0.846 | 0.900 / 0.902 | False → False |
| map@12h | measurement_pure_ridge | 15004 / 1166 | True | 0.856 → 0.860 | 0.867 → 0.872 | 0.902 / 0.899 | False → False |
| o2sat@12h | measurement_pure_ridge | 15006 / 1080 | True | 0.839 → 0.840 | 0.839 → 0.843 | 0.905 / 0.903 | False → False |
| potassium@12h | measurement_pure_ridge | 3254 / 1038 | True | 0.864 → 0.870 | 0.871 → 0.898 | 0.903 / 0.901 | False → False |
| respiratory_rate@12h | measurement_pure_ridge | 15007 / 1051 | True | 0.866 → 0.879 | 0.871 → 0.886 | 0.900 / 0.905 | False → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
