# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `3`
- Point pass: `3`
- Interval pass, fixed → value-conditional: `0` → `0`
- Full pass, fixed → value-conditional: `1` → `1`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 15002 / 5524 | True | 0.872 → 0.891 | 0.785 → 0.839 | 0.901 / 0.902 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15020 / 2161 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| platelets@1h | target_specific_change_hurdle | 2787 / 1814 | True | 0.912 → 0.908 | 0.825 → 0.824 | 0.900 / 0.903 | False → False |
| urine_output@1h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
