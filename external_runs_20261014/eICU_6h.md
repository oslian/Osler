# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `6`
- Interval pass, fixed → value-conditional: `1` → `4`
- Full pass, fixed → value-conditional: `1` → `4`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@6h | target_specific_change_hurdle | 15013 / 3316 | True | 0.732 → 0.833 | 0.758 → 0.837 | 0.899 / 0.927 | False → False |
| glucose@6h | observed_treatment_context_ridge | 13791 / 1072 | False | 0.786 → 0.840 | 0.766 → 0.840 | 0.900 / 0.925 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15000 / 1146 | True | 0.887 → 0.913 | 0.878 → 0.907 | 0.903 / 0.931 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | 3542 / 714 | False | 0.820 → 0.849 | 0.816 → 0.848 | 0.902 / 0.918 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3663 / 721 | True | 0.862 → 0.880 | 0.850 → 0.872 | 0.899 / 0.916 | False → True |
| map@6h | measurement_pure_ridge | 15027 / 1143 | True | 0.843 → 0.871 | 0.842 → 0.867 | 0.904 / 0.912 | False → False |
| potassium@6h | measurement_pure_ridge | 4411 / 909 | True | 0.866 → 0.886 | 0.901 → 0.913 | 0.902 / 0.911 | False → True |
| respiratory_rate@6h | measurement_pure_ridge | 15017 / 1023 | True | 0.861 → 0.891 | 0.876 → 0.903 | 0.902 / 0.919 | False → True |
| urine_output@6h | target_specific_change_hurdle | 15013 / 1155 | False | 0.619 → 0.694 | 0.550 → 0.637 | 0.903 / 0.915 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
