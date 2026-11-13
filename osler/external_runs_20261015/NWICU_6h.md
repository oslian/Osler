# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `6`
- Point pass: `6`
- Interval pass, fixed → value-conditional: `4` → `5`
- Full pass, fixed → value-conditional: `4` → `5`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@6h | target_specific_change_hurdle | 15000 / 5376 | True | 0.877 → 0.876 | 0.903 → 0.918 | 0.899 / 0.899 | True → True |
| glucose@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@6h | measurement_pure_ridge | 15003 / 1996 | True | 0.890 → 0.887 | 0.904 → 0.897 | 0.903 / 0.900 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| hemoglobin@6h | target_specific_change_hurdle | 14808 / 5986 | True | 0.918 → 0.917 | 0.902 → 0.901 | 0.899 / 0.900 | True → True |
| map@6h | measurement_pure_ridge | 15000 / 2043 | True | 0.897 → 0.911 | 0.909 → 0.923 | 0.904 / 0.900 | True → True |
| potassium@6h | measurement_pure_ridge | 15002 / 5145 | True | 0.895 → 0.899 | 0.775 → 0.838 | 0.902 / 0.897 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15008 / 2164 | True | 0.892 → 0.898 | 0.859 → 0.878 | 0.902 / 0.900 | False → True |
| urine_output@6h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
