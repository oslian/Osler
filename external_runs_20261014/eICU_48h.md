# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `5`
- Point pass: `4`
- Interval pass, fixed → value-conditional: `1` → `1`
- Full pass, fixed → value-conditional: `1` → `1`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@48h | target_specific_change_hurdle | 15008 / 1977 | True | 0.797 → 0.915 | 0.807 → 0.923 | 0.901 / 0.920 | False → True |
| glucose@48h | target_specific_change_hurdle | 15005 / 1039 | True | 0.846 → 0.868 | 0.821 → 0.865 | 0.900 / 0.916 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15012 / 784 | True | 0.922 → 0.938 | 0.921 → 0.933 | 0.903 / 0.919 | True → False |
| map@48h | observed_treatment_context_ridge | 15021 / 812 | False | 0.837 → 0.864 | 0.833 → 0.863 | 0.903 / 0.917 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 1785 | True | 0.859 → 0.861 | 0.861 → 0.862 | 0.902 / 0.906 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
