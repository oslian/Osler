# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `5`
- Point pass: `4`
- Interval pass, fixed → value-conditional: `1` → `3`
- Full pass, fixed → value-conditional: `1` → `3`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@48h | target_specific_change_hurdle | 15000 / 4917 | True | 0.802 → 0.874 | 0.801 → 0.881 | 0.901 / 0.903 | False → True |
| glucose@48h | target_specific_change_hurdle | 15003 / 4954 | True | 0.872 → 0.888 | 0.841 → 0.872 | 0.899 / 0.901 | False → True |
| heart_rate@48h | measurement_pure_ridge | 15010 / 910 | True | 0.912 → 0.905 | 0.916 → 0.910 | 0.903 / 0.899 | True → True |
| map@48h | observed_treatment_context_ridge | 15024 / 907 | False | 0.854 → 0.855 | 0.863 → 0.861 | 0.904 / 0.905 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 4180 | True | 0.862 → 0.860 | 0.848 → 0.846 | 0.900 / 0.899 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
