# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `9`
- Interval pass, fixed → value-conditional: `5` → `7`
- Full pass, fixed → value-conditional: `5` → `7`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@24h | target_specific_change_hurdle | 15003 / 6196 | True | 0.894 → 0.899 | 0.884 → 0.887 | 0.902 / 0.901 | True → True |
| bun@24h | target_specific_change_hurdle | 15003 / 6203 | True | 0.882 → 0.877 | 0.832 → 0.834 | 0.900 / 0.898 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15000 / 6178 | True | 0.891 → 0.913 | 0.872 → 0.921 | 0.901 / 0.897 | True → True |
| glucose@24h | measurement_pure_ridge | 15002 / 3612 | True | 0.868 → 0.893 | 0.846 → 0.871 | 0.899 / 0.898 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15005 / 1914 | True | 0.899 → 0.894 | 0.899 → 0.900 | 0.904 / 0.903 | True → True |
| map@24h | measurement_pure_ridge | 15001 / 1984 | True | 0.916 → 0.913 | 0.931 → 0.922 | 0.902 / 0.901 | False → True |
| potassium@24h | measurement_pure_ridge | 15001 / 5921 | True | 0.918 → 0.906 | 0.913 → 0.905 | 0.900 / 0.903 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15005 / 2072 | True | 0.881 → 0.886 | 0.802 → 0.815 | 0.905 / 0.903 | False → False |
| sodium@24h | target_specific_change_hurdle | 15000 / 5571 | True | 0.904 → 0.906 | 0.914 → 0.917 | 0.901 / 0.899 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
