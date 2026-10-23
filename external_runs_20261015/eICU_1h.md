# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `5`; evaluable: `5`
- Point pass: `2`
- Interval pass, fixed → value-conditional: `0` → `1`
- Full pass, fixed → value-conditional: `1` → `1`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 8510 / 1475 | True | 0.831 → 0.856 | 0.812 → 0.864 | 0.901 / 0.900 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15011 / 1063 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | 15002 / 1162 | False | 0.864 → 0.881 | 0.865 → 0.882 | 0.902 / 0.905 | False → False |
| platelets@1h | target_specific_change_hurdle | 282 / 203 | False | 0.842 → 0.814 | 0.794 → 0.828 | 0.900 / 0.901 | False → False |
| urine_output@1h | target_specific_change_hurdle | 15008 / 1507 | False | 0.723 → 0.793 | 0.700 → 0.782 | 0.903 / 0.904 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
