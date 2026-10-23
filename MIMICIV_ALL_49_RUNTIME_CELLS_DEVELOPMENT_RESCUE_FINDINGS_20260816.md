# MIMIC-IV All Runtime Cells Development Rescue Audit

- Evaluated: `49/49` cells
- Point validated: `49`
- Continuous 90% intervals validated: `47`
- Fallback: `0`
- Validation status: `development_rescue_complete_pending_fresh_lockbox`

| Cell | Task | Physiology | Treatment | Hurdle | Selected Source | Interval | Status / Reason |
|---|---|---|---|---|---|---|---|
| glucose@1h | continuous_value | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| hypoxemia_event@1h | binary_event_probability | None | None | None | mimiciv_hypoxemia_event_model | not_applicable_probability | pass |
| map@1h | continuous_value | False | True | False | observed_treatment_context_ridge | validated_90pct | pass |
| platelets@1h | continuous_value_with_change_hurdle | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| urine_output@1h | continuous_value | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| glucose@3h | continuous_value | True | True | False | observed_treatment_context_ridge | validated_90pct | pass |
| heart_rate@3h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| hematocrit@3h | continuous_value_with_change_hurdle | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| hemoglobin@3h | continuous_value_with_change_hurdle | True | False | True | target_specific_change_hurdle | validated_90pct | pass |
| hypoxemia_event@3h | binary_event_probability | None | None | None | mimiciv_hypoxemia_event_model | not_applicable_probability | pass |
| map@3h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| platelets@3h | continuous_value_with_change_hurdle | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| potassium@3h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| respiratory_rate@3h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| urine_output@3h | continuous_value | True | False | True | target_specific_change_hurdle | validated_90pct | pass |
| creatinine@6h | continuous_value | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| glucose@6h | continuous_value | False | True | False | observed_treatment_context_ridge | validated_90pct | pass |
| heart_rate@6h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| hematocrit@6h | continuous_value_with_change_hurdle | False | True | False | observed_treatment_context_ridge | validated_90pct | pass |
| hemoglobin@6h | continuous_value_with_change_hurdle | True | False | True | target_specific_change_hurdle | validated_90pct | pass |
| map@6h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| potassium@6h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| respiratory_rate@6h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| urine_output@6h | continuous_value | True | False | True | target_specific_change_hurdle | validated_90pct | pass |
| bicarbonate@12h | continuous_value | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| chloride@12h | continuous_value_with_change_hurdle | True | True | True | target_specific_change_hurdle | validated_90pct | pass |
| creatinine@12h | continuous_value | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| glucose@12h | continuous_value | True | True | False | observed_treatment_context_ridge | validated_90pct | pass |
| heart_rate@12h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| hematocrit@12h | continuous_value_with_change_hurdle | True | False | True | target_specific_change_hurdle | validated_90pct | pass |
| hemoglobin@12h | continuous_value_with_change_hurdle | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| map@12h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| o2sat@12h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| potassium@12h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| respiratory_rate@12h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| bicarbonate@24h | continuous_value_with_change_hurdle | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| bun@24h | continuous_value_with_change_hurdle | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| creatinine@24h | continuous_value_with_change_hurdle | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| glucose@24h | continuous_value | True | True | False | measurement_pure_ridge | validated_90pct | pass |
| heart_rate@24h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| map@24h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| potassium@24h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| respiratory_rate@24h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| sodium@24h | continuous_value | False | True | True | target_specific_change_hurdle | validated_90pct | pass |
| creatinine@48h | continuous_value | False | False | True | target_specific_change_hurdle | validated_90pct | pass |
| glucose@48h | continuous_value | True | True | True | target_specific_change_hurdle | validated_90pct | pass |
| heart_rate@48h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |
| map@48h | continuous_value | True | True | False | observed_treatment_context_ridge | validated_90pct | pass |
| potassium@48h | continuous_value | True | False | False | measurement_pure_ridge | validated_90pct | pass |

## Boundary

Target-specific rescue methods were selected after inspecting the original MIMIC care-unit/time failures. All current development gates pass, but these cells require a fresh untouched lockbox before formal external promotion.

This is a factual development audit with care-unit and forward-time stress tests. A pass does not authorize causal, counterfactual, clinical, or treatment-recommendation claims.
