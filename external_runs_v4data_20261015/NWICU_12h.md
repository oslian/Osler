# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `11`; evaluable: `10`
- Point pass: `10`
- Interval pass, fixed → value-conditional: `7` → `6`
- Full pass, fixed → value-conditional: `7` → `6`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@12h | target_specific_change_hurdle | 15001 / 5376 | True | 0.871 → 0.873 | 0.873 → 0.882 | 0.902 / 0.897 | True → True |
| chloride@12h | target_specific_change_hurdle | 15000 / 5308 | True | 0.898 → 0.896 | 0.827 → 0.826 | 0.902 / 0.902 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15010 / 5277 | True | 0.870 → 0.870 | 0.863 → 0.880 | 0.900 / 0.900 | False → False |
| glucose@12h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@12h | measurement_pure_ridge | 15007 / 2122 | True | 0.885 → 0.886 | 0.919 → 0.919 | 0.905 / 0.902 | True → True |
| hematocrit@12h | target_specific_change_hurdle | 11888 / 4656 | True | 0.878 → 0.880 | 0.870 → 0.867 | 0.901 / 0.897 | True → False |
| hemoglobin@12h | measurement_pure_ridge | 12111 / 4753 | True | 0.914 → 0.911 | 0.879 → 0.853 | 0.900 / 0.902 | True → False |
| map@12h | measurement_pure_ridge | 15003 / 2086 | True | 0.902 → 0.903 | 0.891 → 0.894 | 0.902 / 0.899 | True → True |
| o2sat@12h | measurement_pure_ridge | 15004 / 2120 | True | 0.893 → 0.898 | 0.907 → 0.906 | 0.905 / 0.903 | True → True |
| potassium@12h | measurement_pure_ridge | 15001 / 5040 | True | 0.912 → 0.913 | 0.918 → 0.907 | 0.903 / 0.901 | True → True |
| respiratory_rate@12h | measurement_pure_ridge | 15006 / 2206 | True | 0.888 → 0.898 | 0.867 → 0.875 | 0.900 / 0.905 | False → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
