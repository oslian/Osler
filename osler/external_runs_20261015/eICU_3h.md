# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `10`; evaluable: `10`
- Point pass: `3`
- Interval pass, fixed → value-conditional: `4` → `5`
- Full pass, fixed → value-conditional: `2` → `3`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@3h | observed_treatment_context_ridge | 15004 / 1475 | False | 0.824 → 0.842 | 0.778 → 0.830 | 0.899 / 0.902 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15015 / 1048 | False | 0.915 → 0.916 | 0.907 → 0.911 | 0.904 / 0.901 | False → False |
| hematocrit@3h | measurement_pure_ridge | 3104 / 917 | False | 0.875 → 0.871 | 0.904 → 0.900 | 0.900 / 0.899 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 3439 / 962 | False | 0.860 → 0.851 | 0.852 → 0.837 | 0.900 / 0.901 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15016 / 968 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15010 / 1051 | True | 0.850 → 0.872 | 0.851 → 0.873 | 0.902 / 0.900 | False → True |
| platelets@3h | target_specific_change_hurdle | 1117 / 540 | False | 0.823 → 0.800 | 0.850 → 0.834 | 0.901 / 0.902 | False → False |
| potassium@3h | measurement_pure_ridge | 3411 / 1066 | False | 0.873 → 0.875 | 0.893 → 0.893 | 0.903 / 0.901 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15006 / 941 | True | 0.891 → 0.909 | 0.893 → 0.909 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15007 / 1216 | False | 0.687 → 0.789 | 0.631 → 0.749 | 0.905 / 0.904 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
