# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `10`; evaluable: `8`
- Point pass: `8`
- Interval pass, fixed → value-conditional: `7` → `7`
- Full pass, fixed → value-conditional: `8` → `8`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@3h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1980 | True | 0.909 → 0.908 | 0.906 → 0.911 | 0.903 / 0.902 | True → True |
| hematocrit@3h | measurement_pure_ridge | 10458 / 4549 | True | 0.916 → 0.917 | 0.891 → 0.894 | 0.900 / 0.900 | True → True |
| hemoglobin@3h | target_specific_change_hurdle | 10828 / 4727 | True | 0.912 → 0.911 | 0.897 → 0.902 | 0.899 / 0.901 | True → True |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15002 / 2026 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15004 / 2001 | True | 0.913 → 0.921 | 0.905 → 0.900 | 0.900 / 0.899 | True → True |
| platelets@3h | target_specific_change_hurdle | 9051 / 4100 | True | 0.907 → 0.903 | 0.926 → 0.921 | 0.900 / 0.902 | True → True |
| potassium@3h | measurement_pure_ridge | 15014 / 5690 | True | 0.909 → 0.905 | 0.907 → 0.897 | 0.902 / 0.903 | True → True |
| respiratory_rate@3h | measurement_pure_ridge | 15002 / 2117 | True | 0.897 → 0.909 | 0.877 → 0.891 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
