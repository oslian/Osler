# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `6`
- Interval pass, fixed → value-conditional: `1` → `3`
- Full pass, fixed → value-conditional: `1` → `3`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@6h | target_specific_change_hurdle | 15013 / 3316 | True | 0.732 → 0.759 | 0.758 → 0.776 | 0.899 / 0.899 | False → False |
| glucose@6h | observed_treatment_context_ridge | 13791 / 1072 | False | 0.786 → 0.802 | 0.766 → 0.807 | 0.900 / 0.899 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15000 / 1146 | True | 0.887 → 0.887 | 0.878 → 0.884 | 0.903 / 0.900 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | 3542 / 714 | False | 0.820 → 0.816 | 0.816 → 0.812 | 0.902 / 0.898 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3663 / 721 | True | 0.862 → 0.860 | 0.850 → 0.842 | 0.899 / 0.900 | False → False |
| map@6h | measurement_pure_ridge | 15027 / 1143 | True | 0.843 → 0.864 | 0.842 → 0.860 | 0.904 / 0.900 | False → False |
| potassium@6h | measurement_pure_ridge | 4411 / 909 | True | 0.866 → 0.877 | 0.901 → 0.899 | 0.902 / 0.897 | False → True |
| respiratory_rate@6h | measurement_pure_ridge | 15017 / 1023 | True | 0.861 → 0.879 | 0.876 → 0.894 | 0.902 / 0.900 | False → True |
| urine_output@6h | target_specific_change_hurdle | 15013 / 1155 | False | 0.619 → 0.687 | 0.550 → 0.627 | 0.903 / 0.903 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
