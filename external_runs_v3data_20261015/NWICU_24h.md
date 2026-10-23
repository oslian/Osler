# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `9`
- Interval pass, fixed → value-conditional: `6` → `6`
- Full pass, fixed → value-conditional: `6` → `6`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@24h | target_specific_change_hurdle | 15003 / 6196 | True | 0.892 → 0.898 | 0.883 → 0.887 | 0.902 / 0.900 | True → True |
| bun@24h | target_specific_change_hurdle | 15003 / 6203 | True | 0.882 → 0.875 | 0.833 → 0.832 | 0.901 / 0.900 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15000 / 6178 | True | 0.893 → 0.912 | 0.873 → 0.920 | 0.903 / 0.899 | True → True |
| glucose@24h | measurement_pure_ridge | 15002 / 3612 | True | 0.867 → 0.892 | 0.849 → 0.865 | 0.899 / 0.899 | False → False |
| heart_rate@24h | measurement_pure_ridge | 15005 / 1914 | True | 0.898 → 0.894 | 0.899 → 0.900 | 0.904 / 0.904 | True → True |
| map@24h | measurement_pure_ridge | 15001 / 1984 | True | 0.916 → 0.912 | 0.929 → 0.921 | 0.900 / 0.900 | True → True |
| potassium@24h | measurement_pure_ridge | 15001 / 5921 | True | 0.916 → 0.904 | 0.912 → 0.903 | 0.900 / 0.900 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15005 / 2072 | True | 0.881 → 0.887 | 0.802 → 0.816 | 0.906 / 0.901 | False → False |
| sodium@24h | target_specific_change_hurdle | 15000 / 5571 | True | 0.904 → 0.904 | 0.904 → 0.917 | 0.900 / 0.900 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
