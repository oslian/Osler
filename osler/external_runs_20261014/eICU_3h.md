# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `10`; evaluable: `10`
- Point pass: `3`
- Interval pass, fixed → value-conditional: `4` → `5`
- Full pass, fixed → value-conditional: `2` → `3`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@3h | observed_treatment_context_ridge | 15004 / 1475 | False | 0.824 → 0.856 | 0.778 → 0.849 | 0.899 / 0.909 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15015 / 1048 | False | 0.915 → 0.929 | 0.907 → 0.923 | 0.904 / 0.923 | False → False |
| hematocrit@3h | measurement_pure_ridge | 3104 / 917 | False | 0.875 → 0.892 | 0.904 → 0.913 | 0.900 / 0.919 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 3439 / 962 | False | 0.860 → 0.865 | 0.852 → 0.857 | 0.900 / 0.912 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15016 / 968 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15010 / 1051 | True | 0.850 → 0.882 | 0.851 → 0.884 | 0.902 / 0.916 | False → True |
| platelets@3h | target_specific_change_hurdle | 1117 / 540 | False | 0.823 → 0.823 | 0.850 → 0.838 | 0.901 / 0.912 | False → False |
| potassium@3h | measurement_pure_ridge | 3411 / 1066 | False | 0.873 → 0.892 | 0.893 → 0.904 | 0.903 / 0.913 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15006 / 941 | True | 0.891 → 0.906 | 0.893 → 0.907 | 0.904 / 0.906 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15007 / 1216 | False | 0.687 → 0.726 | 0.631 → 0.670 | 0.905 / 0.907 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
