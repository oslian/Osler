# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `6`
- Point pass: `6`
- Interval pass, fixed → value-conditional: `4` → `5`
- Full pass, fixed → value-conditional: `4` → `5`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@6h | target_specific_change_hurdle | 15000 / 5376 | True | 0.879 → 0.873 | 0.905 → 0.916 | 0.897 / 0.898 | True → True |
| glucose@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@6h | measurement_pure_ridge | 15003 / 1996 | True | 0.891 → 0.887 | 0.905 → 0.897 | 0.902 / 0.902 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| hemoglobin@6h | target_specific_change_hurdle | 14808 / 5986 | True | 0.915 → 0.915 | 0.900 → 0.900 | 0.899 / 0.901 | True → True |
| map@6h | measurement_pure_ridge | 15000 / 2043 | True | 0.896 → 0.912 | 0.910 → 0.924 | 0.902 / 0.899 | True → True |
| potassium@6h | measurement_pure_ridge | 15002 / 5145 | True | 0.895 → 0.903 | 0.757 → 0.833 | 0.901 / 0.901 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15008 / 2164 | True | 0.891 → 0.901 | 0.858 → 0.886 | 0.903 / 0.898 | False → True |
| urine_output@6h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
