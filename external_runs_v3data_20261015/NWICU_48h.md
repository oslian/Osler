# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `4`
- Point pass: `4`
- Interval pass, fixed → value-conditional: `2` → `3`
- Full pass, fixed → value-conditional: `2` → `3`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| creatinine@48h | target_specific_change_hurdle | 14326 / 5904 | True | 0.883 → 0.903 | 0.856 → 0.884 | 0.900 / 0.901 | False → True |
| glucose@48h | target_specific_change_hurdle | 15001 / 3624 | True | 0.838 → 0.856 | 0.836 → 0.872 | 0.899 / 0.898 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15006 / 1852 | True | 0.908 → 0.907 | 0.921 → 0.921 | 0.902 / 0.896 | True → True |
| map@48h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| potassium@48h | measurement_pure_ridge | 15001 / 5865 | True | 0.909 → 0.904 | 0.917 → 0.895 | 0.901 / 0.899 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
