# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `9`; evaluable: `9`
- Point pass: `9`
- Interval pass, fixed → value-conditional: `5` → `8`
- Full pass, fixed → value-conditional: `5` → `8`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| bicarbonate@24h | target_specific_change_hurdle | 15004 / 1792 | True | 0.871 → 0.884 | 0.878 → 0.887 | 0.902 / 0.906 | True → True |
| bun@24h | target_specific_change_hurdle | 15005 / 2057 | True | 0.793 → 0.888 | 0.801 → 0.890 | 0.900 / 0.916 | False → True |
| creatinine@24h | target_specific_change_hurdle | 15008 / 2047 | True | 0.778 → 0.895 | 0.787 → 0.892 | 0.901 / 0.923 | False → True |
| glucose@24h | measurement_pure_ridge | 15012 / 1084 | True | 0.864 → 0.880 | 0.865 → 0.892 | 0.899 / 0.909 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15006 / 866 | True | 0.896 → 0.943 | 0.906 → 0.945 | 0.904 / 0.944 | True → False |
| map@24h | measurement_pure_ridge | 15012 / 849 | True | 0.897 → 0.895 | 0.900 → 0.899 | 0.902 / 0.911 | True → True |
| potassium@24h | measurement_pure_ridge | 15014 / 1828 | True | 0.887 → 0.888 | 0.884 → 0.886 | 0.900 / 0.908 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15012 / 870 | True | 0.864 → 0.880 | 0.860 → 0.875 | 0.905 / 0.913 | False → True |
| sodium@24h | target_specific_change_hurdle | 15003 / 1963 | True | 0.900 → 0.920 | 0.892 → 0.917 | 0.901 / 0.911 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
