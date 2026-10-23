# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `5`
- Point pass: `4`
- Interval pass, fixed → value-conditional: `1` → `2`
- Full pass, fixed → value-conditional: `1` → `2`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@48h | target_specific_change_hurdle | 15000 / 4917 | True | 0.799 → 0.874 | 0.798 → 0.880 | 0.900 / 0.901 | False → True |
| glucose@48h | target_specific_change_hurdle | 15003 / 4954 | True | 0.860 → 0.874 | 0.828 → 0.857 | 0.899 / 0.898 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15010 / 910 | True | 0.910 → 0.910 | 0.915 → 0.913 | 0.902 / 0.896 | True → True |
| map@48h | observed_treatment_context_ridge | 15024 / 907 | False | 0.846 → 0.852 | 0.859 → 0.859 | 0.904 / 0.905 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 4180 | True | 0.863 → 0.859 | 0.850 → 0.844 | 0.901 / 0.899 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
