# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `10`; evaluable: `8`
- Point pass: `8`
- Interval pass, fixed → value-conditional: `7` → `5`
- Full pass, fixed → value-conditional: `8` → `6`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@3h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1980 | True | 0.908 → 0.922 | 0.906 → 0.935 | 0.904 / 0.923 | True → False |
| hematocrit@3h | measurement_pure_ridge | 10458 / 4549 | True | 0.916 → 0.933 | 0.891 → 0.904 | 0.900 / 0.919 | True → False |
| hemoglobin@3h | target_specific_change_hurdle | 10828 / 4727 | True | 0.913 → 0.920 | 0.899 → 0.906 | 0.900 / 0.912 | True → True |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15002 / 2026 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15004 / 2001 | True | 0.914 → 0.929 | 0.905 → 0.920 | 0.902 / 0.916 | True → True |
| platelets@3h | target_specific_change_hurdle | 9051 / 4100 | True | 0.907 → 0.915 | 0.925 → 0.928 | 0.901 / 0.912 | True → True |
| potassium@3h | measurement_pure_ridge | 15014 / 5690 | True | 0.911 → 0.916 | 0.917 → 0.920 | 0.903 / 0.913 | True → True |
| respiratory_rate@3h | measurement_pure_ridge | 15002 / 2117 | True | 0.896 → 0.907 | 0.873 → 0.884 | 0.904 / 0.906 | True → True |
| urine_output@3h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
