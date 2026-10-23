# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `5`
- Point pass: `2`
- Interval pass, fixed → value-conditional: `1` → `1`
- Full pass, fixed → value-conditional: `2` → `1`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 333 / 196 | False | 0.657 → 0.769 | 0.571 → 0.760 | 0.902 / 0.899 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15025 / 1204 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | 15024 / 1255 | False | 0.858 → 0.879 | 0.864 → 0.886 | 0.904 / 0.903 | False → False |
| platelets@1h | target_specific_change_hurdle | 226 / 176 | False | 0.854 → 0.820 | 0.839 → 0.855 | 0.900 / 0.902 | False → False |
| urine_output@1h | target_specific_change_hurdle | 15016 / 1497 | True | 0.885 → 0.863 | 0.906 → 0.870 | 0.905 / 0.903 | True → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
