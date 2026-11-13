# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `9`
- Interval pass, fixed → value-conditional: `5` → `5`
- Full pass, fixed → value-conditional: `5` → `5`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@24h | target_specific_change_hurdle | 15003 / 6196 | True | 0.894 → 0.903 | 0.884 → 0.906 | 0.902 / 0.906 | True → True |
| bun@24h | target_specific_change_hurdle | 15003 / 6203 | True | 0.882 → 0.897 | 0.832 → 0.920 | 0.900 / 0.916 | False → True |
| creatinine@24h | target_specific_change_hurdle | 15000 / 6178 | True | 0.891 → 0.931 | 0.872 → 0.939 | 0.901 / 0.923 | True → False |
| glucose@24h | measurement_pure_ridge | 15002 / 3612 | True | 0.868 → 0.899 | 0.846 → 0.873 | 0.899 / 0.909 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15005 / 1914 | True | 0.899 → 0.941 | 0.899 → 0.938 | 0.904 / 0.944 | True → False |
| map@24h | measurement_pure_ridge | 15001 / 1984 | True | 0.916 → 0.922 | 0.931 → 0.934 | 0.902 / 0.911 | False → False |
| potassium@24h | measurement_pure_ridge | 15001 / 5921 | True | 0.918 → 0.915 | 0.913 → 0.912 | 0.900 / 0.908 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15005 / 2072 | True | 0.881 → 0.894 | 0.802 → 0.823 | 0.905 / 0.913 | False → False |
| sodium@24h | target_specific_change_hurdle | 15000 / 5571 | True | 0.904 → 0.920 | 0.914 → 0.928 | 0.901 / 0.911 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
