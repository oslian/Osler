# MIMIC-IV to eICU 49-Cell External Validation

- Catalog: `49`
- Evaluable: `49`
- Point pass: `36`
- Continuous interval pass: `12`
- Full external pass: `11`
- Not evaluable: `0`

| Cell | MIMIC source | eICU module | Rows / hospitals | Point | Interval | Full | Failure |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | endocrine_stress | 8510 / 22 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | respiratory | 15011 / 161 | True | None | True | - |
| map@1h | observed_treatment_context_ridge | cardiovascular_instability | 15002 / 12 | False | False | False | does_not_beat_measurement_pure_ridge_hospital, does_not_beat_measurement_pure_ridge_patient, hospital_interval_miscalibration, patient_interval_miscalibration |
| platelets@1h | target_specific_change_hurdle | coagulopathy_heme | 282 / 17 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| urine_output@1h | target_specific_change_hurdle | aki | 15008 / 123 | False | False | False | does_not_beat_plain_ridge_hospital, hospital_interval_miscalibration, patient_interval_miscalibration |
| glucose@3h | observed_treatment_context_ridge | endocrine_stress | 15004 / 23 | False | False | False | does_not_beat_measurement_pure_ridge_hospital, does_not_beat_measurement_pure_ridge_patient, hospital_interval_miscalibration, patient_interval_miscalibration |
| heart_rate@3h | measurement_pure_ridge | cardiovascular_instability | 15015 / 12 | False | True | False | does_not_beat_persistence_hospital, does_not_beat_persistence_matched_baseline_hospital, does_not_beat_persistence_matched_baseline_patient, does_not_beat_persistence_patient |
| hematocrit@3h | measurement_pure_ridge | coagulopathy_heme | 3104 / 23 | False | True | False | does_not_beat_persistence_hospital, does_not_beat_persistence_matched_baseline_hospital |
| hemoglobin@3h | target_specific_change_hurdle | coagulopathy_heme | 3439 / 24 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | respiratory | 15016 / 162 | True | None | True | - |
| map@3h | measurement_pure_ridge | cardiovascular_instability | 15010 / 12 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| platelets@3h | target_specific_change_hurdle | coagulopathy_heme | 1117 / 21 | False | False | False | does_not_beat_plain_ridge_hospital, hospital_interval_miscalibration, patient_interval_miscalibration |
| potassium@3h | measurement_pure_ridge | electrolyte_acid_base | 3411 / 32 | False | True | False | does_not_beat_persistence_hospital, does_not_beat_persistence_matched_baseline_hospital |
| respiratory_rate@3h | measurement_pure_ridge | respiratory | 15006 / 159 | True | True | True | - |
| urine_output@3h | target_specific_change_hurdle | aki | 15007 / 128 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| creatinine@6h | target_specific_change_hurdle | aki | 15013 / 177 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| glucose@6h | observed_treatment_context_ridge | endocrine_stress | 13791 / 12 | False | False | False | does_not_beat_measurement_pure_ridge_hospital, does_not_beat_measurement_pure_ridge_patient, hospital_interval_miscalibration, patient_interval_miscalibration |
| heart_rate@6h | measurement_pure_ridge | cardiovascular_instability | 15000 / 12 | True | True | True | - |
| hematocrit@6h | observed_treatment_context_ridge | coagulopathy_heme | 3542 / 12 | False | False | False | does_not_beat_measurement_pure_ridge_hospital, does_not_beat_measurement_pure_ridge_patient, hospital_interval_miscalibration, patient_interval_miscalibration |
| hemoglobin@6h | target_specific_change_hurdle | coagulopathy_heme | 3663 / 12 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| map@6h | measurement_pure_ridge | cardiovascular_instability | 15027 / 12 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| potassium@6h | measurement_pure_ridge | electrolyte_acid_base | 4411 / 26 | True | False | False | patient_interval_miscalibration |
| respiratory_rate@6h | measurement_pure_ridge | respiratory | 15017 / 49 | True | False | False | patient_interval_miscalibration |
| urine_output@6h | target_specific_change_hurdle | aki | 15013 / 133 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| bicarbonate@12h | target_specific_change_hurdle | electrolyte_acid_base | 7008 / 34 | False | False | False | does_not_beat_plain_ridge_hospital, hospital_interval_miscalibration, patient_interval_miscalibration |
| chloride@12h | target_specific_change_hurdle | electrolyte_acid_base | 4707 / 32 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| creatinine@12h | target_specific_change_hurdle | aki | 15004 / 177 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| glucose@12h | observed_treatment_context_ridge | endocrine_stress | 15006 / 22 | False | False | False | does_not_beat_measurement_pure_ridge_hospital, does_not_beat_measurement_pure_ridge_patient, hospital_interval_miscalibration, patient_interval_miscalibration |
| heart_rate@12h | measurement_pure_ridge | cardiovascular_instability | 15018 / 12 | True | False | False | hospital_interval_miscalibration |
| hematocrit@12h | target_specific_change_hurdle | coagulopathy_heme | 6787 / 24 | False | False | False | does_not_beat_plain_ridge_hospital, hospital_interval_miscalibration, patient_interval_miscalibration |
| hemoglobin@12h | measurement_pure_ridge | coagulopathy_heme | 7263 / 24 | True | True | True | - |
| map@12h | measurement_pure_ridge | cardiovascular_instability | 15000 / 12 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| o2sat@12h | measurement_pure_ridge | respiratory | 15010 / 159 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| potassium@12h | measurement_pure_ridge | electrolyte_acid_base | 7533 / 33 | True | True | True | - |
| respiratory_rate@12h | measurement_pure_ridge | respiratory | 15023 / 157 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| bicarbonate@24h | target_specific_change_hurdle | aki | 15004 / 164 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| bun@24h | target_specific_change_hurdle | aki | 15005 / 174 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| creatinine@24h | target_specific_change_hurdle | aki | 15008 / 174 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| glucose@24h | measurement_pure_ridge | aki | 15012 / 154 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| heart_rate@24h | measurement_pure_ridge | aki | 15006 / 154 | True | True | True | - |
| map@24h | measurement_pure_ridge | aki | 15012 / 150 | True | True | True | - |
| potassium@24h | measurement_pure_ridge | aki | 15014 / 172 | True | True | True | - |
| respiratory_rate@24h | measurement_pure_ridge | aki | 15012 / 153 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| sodium@24h | target_specific_change_hurdle | aki | 15003 / 167 | True | True | True | - |
| creatinine@48h | target_specific_change_hurdle | aki | 15008 / 165 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| glucose@48h | target_specific_change_hurdle | aki | 15005 / 150 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |
| heart_rate@48h | measurement_pure_ridge | aki | 15012 / 144 | True | True | True | - |
| map@48h | observed_treatment_context_ridge | aki | 15021 / 151 | False | False | False | does_not_beat_measurement_pure_ridge_hospital, does_not_beat_measurement_pure_ridge_patient, hospital_interval_miscalibration, patient_interval_miscalibration |
| potassium@48h | measurement_pure_ridge | aki | 15002 / 166 | True | False | False | hospital_interval_miscalibration, patient_interval_miscalibration |

## Boundary

All model fitting, source selection, clipping, and conformal calibration use MIMIC-IV only. eICU labels are used only for final scoring; no eICU rescue or retuning is allowed.

eICU is an independent database for these MIMIC-fitted recipes, but it is not a pristine project-wide never-seen lockbox because earlier Osler experiments used eICU.

All outputs remain factual research-only: `causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
