# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `10`; evaluable: `10`
- Point pass: `5`
- Interval pass, fixed → value-conditional: `3` → `5`
- Full pass, fixed → value-conditional: `3` → `5`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@3h | observed_treatment_context_ridge | 1022 / 452 | False | 0.728 → 0.785 | 0.681 → 0.754 | 0.900 / 0.902 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1108 | True | 0.900 → 0.897 | 0.886 → 0.888 | 0.903 / 0.900 | True → True |
| hematocrit@3h | measurement_pure_ridge | 2637 / 861 | False | 0.864 → 0.866 | 0.852 → 0.881 | 0.900 / 0.900 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 2918 / 904 | False | 0.858 → 0.858 | 0.875 → 0.863 | 0.901 / 0.902 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15014 / 1027 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15005 / 1113 | True | 0.853 → 0.874 | 0.859 → 0.879 | 0.901 / 0.898 | False → True |
| platelets@3h | target_specific_change_hurdle | 954 / 486 | False | 0.827 → 0.808 | 0.850 → 0.829 | 0.901 / 0.901 | False → False |
| potassium@3h | measurement_pure_ridge | 2886 / 994 | False | 0.871 → 0.877 | 0.905 → 0.905 | 0.899 / 0.900 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15016 / 1040 | True | 0.894 → 0.907 | 0.890 → 0.903 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15004 / 1255 | True | 0.903 → 0.906 | 0.868 → 0.886 | 0.903 / 0.903 | False → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
