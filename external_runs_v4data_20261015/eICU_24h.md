# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `8`
- Interval pass, fixed → value-conditional: `5` → `5`
- Full pass, fixed → value-conditional: `5` → `5`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@24h | target_specific_change_hurdle | 15001 / 3977 | True | 0.855 → 0.867 | 0.848 → 0.861 | 0.899 / 0.899 | False → False |
| bun@24h | target_specific_change_hurdle | 15000 / 4917 | True | 0.790 → 0.835 | 0.796 → 0.843 | 0.903 / 0.904 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15000 / 4978 | True | 0.779 → 0.864 | 0.787 → 0.885 | 0.903 / 0.899 | False → False |
| glucose@24h | measurement_pure_ridge | 15007 / 4851 | False | 0.842 → 0.816 | 0.819 → 0.796 | 0.900 / 0.900 | False → False |
| heart_rate@24h | measurement_pure_ridge | 15017 / 925 | True | 0.883 → 0.887 | 0.889 → 0.897 | 0.904 / 0.903 | True → True |
| map@24h | measurement_pure_ridge | 15008 / 893 | True | 0.909 → 0.895 | 0.909 → 0.885 | 0.901 / 0.901 | True → True |
| potassium@24h | measurement_pure_ridge | 15001 / 4266 | True | 0.873 → 0.871 | 0.875 → 0.876 | 0.899 / 0.901 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15017 / 933 | True | 0.891 → 0.891 | 0.908 → 0.903 | 0.905 / 0.901 | True → True |
| sodium@24h | target_specific_change_hurdle | 15002 / 4459 | True | 0.900 → 0.903 | 0.897 → 0.902 | 0.901 / 0.900 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
