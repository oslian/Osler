# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `6`
- Interval pass, fixed → value-conditional: `2` → `2`
- Full pass, fixed → value-conditional: `2` → `2`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@6h | target_specific_change_hurdle | 15009 / 4527 | True | 0.745 → 0.766 | 0.742 → 0.776 | 0.898 / 0.897 | False → False |
| glucose@6h | observed_treatment_context_ridge | 1483 / 630 | False | 0.699 → 0.760 | 0.584 → 0.693 | 0.902 / 0.897 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15006 / 1090 | True | 0.875 → 0.875 | 0.855 → 0.861 | 0.903 / 0.902 | False → False |
| hematocrit@6h | observed_treatment_context_ridge | 3585 / 1017 | False | 0.821 → 0.814 | 0.849 → 0.828 | 0.902 / 0.899 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3950 / 1068 | False | 0.840 → 0.837 | 0.850 → 0.848 | 0.901 / 0.901 | False → False |
| map@6h | measurement_pure_ridge | 15026 / 1127 | True | 0.849 → 0.870 | 0.847 → 0.867 | 0.902 / 0.903 | False → False |
| potassium@6h | measurement_pure_ridge | 3845 / 1125 | True | 0.844 → 0.861 | 0.852 → 0.871 | 0.902 / 0.897 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15004 / 1086 | True | 0.879 → 0.893 | 0.872 → 0.890 | 0.902 / 0.899 | True → True |
| urine_output@6h | target_specific_change_hurdle | 15006 / 1210 | True | 0.914 → 0.918 | 0.899 → 0.905 | 0.903 / 0.904 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
