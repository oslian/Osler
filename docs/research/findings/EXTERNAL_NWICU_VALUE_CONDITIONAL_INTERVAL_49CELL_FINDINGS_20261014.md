# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `49`; evaluable: `40`
- Point pass: `40`
- Interval pass, fixed → value-conditional: `24` → `24`
- Full pass, fixed → value-conditional: `26` → `26`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 15002 / 5524 | True | 0.872 → 0.891 | 0.785 → 0.839 | 0.901 / 0.902 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15020 / 2161 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| platelets@1h | target_specific_change_hurdle | 2787 / 1814 | True | 0.912 → 0.908 | 0.825 → 0.824 | 0.900 / 0.903 | False → False |
| urine_output@1h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |
| glucose@3h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1980 | True | 0.908 → 0.922 | 0.906 → 0.935 | 0.904 / 0.923 | True → False |
| hematocrit@3h | measurement_pure_ridge | 10458 / 4549 | True | 0.916 → 0.933 | 0.891 → 0.904 | 0.900 / 0.919 | True → False |
| hemoglobin@3h | target_specific_change_hurdle | 10828 / 4727 | True | 0.913 → 0.920 | 0.899 → 0.906 | 0.900 / 0.912 | True → True |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15002 / 2026 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15004 / 2001 | True | 0.914 → 0.929 | 0.905 → 0.920 | 0.902 / 0.916 | True → True |
| platelets@3h | target_specific_change_hurdle | 9051 / 4100 | True | 0.907 → 0.915 | 0.925 → 0.928 | 0.901 / 0.912 | True → True |
| potassium@3h | measurement_pure_ridge | 15014 / 5690 | True | 0.911 → 0.916 | 0.917 → 0.920 | 0.903 / 0.913 | True → True |
| respiratory_rate@3h | measurement_pure_ridge | 15002 / 2117 | True | 0.896 → 0.907 | 0.873 → 0.884 | 0.904 / 0.906 | True → True |
| urine_output@3h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |
| creatinine@6h | target_specific_change_hurdle | 15000 / 5376 | True | 0.877 → 0.903 | 0.903 → 0.915 | 0.899 / 0.927 | True → True |
| glucose@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@6h | measurement_pure_ridge | 15003 / 1996 | True | 0.890 → 0.918 | 0.904 → 0.927 | 0.903 / 0.931 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| hemoglobin@6h | target_specific_change_hurdle | 14808 / 5986 | True | 0.918 → 0.932 | 0.902 → 0.930 | 0.899 / 0.916 | True → False |
| map@6h | measurement_pure_ridge | 15000 / 2043 | True | 0.897 → 0.914 | 0.909 → 0.926 | 0.904 / 0.912 | True → True |
| potassium@6h | measurement_pure_ridge | 15002 / 5145 | True | 0.895 → 0.906 | 0.775 → 0.843 | 0.902 / 0.911 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15008 / 2164 | True | 0.892 → 0.910 | 0.859 → 0.904 | 0.902 / 0.919 | False → True |
| urine_output@6h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |
| bicarbonate@12h | target_specific_change_hurdle | 15001 / 5376 | True | 0.871 → 0.890 | 0.873 → 0.910 | 0.900 / 0.917 | True → True |
| chloride@12h | target_specific_change_hurdle | 15000 / 5308 | True | 0.899 → 0.908 | 0.829 → 0.840 | 0.902 / 0.912 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15010 / 5277 | True | 0.871 → 0.899 | 0.863 → 0.911 | 0.900 / 0.930 | False → True |
| glucose@12h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@12h | measurement_pure_ridge | 15007 / 2122 | True | 0.883 → 0.925 | 0.917 → 0.949 | 0.903 / 0.939 | True → False |
| hematocrit@12h | target_specific_change_hurdle | 11888 / 4656 | True | 0.879 → 0.912 | 0.871 → 0.945 | 0.900 / 0.926 | True → False |
| hemoglobin@12h | measurement_pure_ridge | 12111 / 4753 | True | 0.915 → 0.928 | 0.857 → 0.893 | 0.900 / 0.923 | False → True |
| map@12h | measurement_pure_ridge | 15003 / 2086 | True | 0.902 → 0.917 | 0.891 → 0.906 | 0.905 / 0.915 | True → True |
| o2sat@12h | measurement_pure_ridge | 15004 / 2120 | True | 0.894 → 0.906 | 0.908 → 0.915 | 0.906 / 0.920 | True → True |
| potassium@12h | measurement_pure_ridge | 15001 / 5040 | True | 0.912 → 0.914 | 0.918 → 0.908 | 0.904 / 0.902 | True → True |
| respiratory_rate@12h | measurement_pure_ridge | 15006 / 2206 | True | 0.890 → 0.897 | 0.868 → 0.873 | 0.899 / 0.911 | False → True |
| bicarbonate@24h | target_specific_change_hurdle | 15003 / 6196 | True | 0.894 → 0.903 | 0.884 → 0.906 | 0.902 / 0.906 | True → True |
| bun@24h | target_specific_change_hurdle | 15003 / 6203 | True | 0.882 → 0.897 | 0.832 → 0.920 | 0.900 / 0.916 | False → True |
| creatinine@24h | target_specific_change_hurdle | 15000 / 6178 | True | 0.891 → 0.931 | 0.872 → 0.939 | 0.901 / 0.923 | True → False |
| glucose@24h | measurement_pure_ridge | 15002 / 3612 | True | 0.868 → 0.899 | 0.846 → 0.873 | 0.899 / 0.909 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15005 / 1914 | True | 0.899 → 0.941 | 0.899 → 0.938 | 0.904 / 0.944 | True → False |
| map@24h | measurement_pure_ridge | 15001 / 1984 | True | 0.916 → 0.922 | 0.931 → 0.934 | 0.902 / 0.911 | False → False |
| potassium@24h | measurement_pure_ridge | 15001 / 5921 | True | 0.918 → 0.915 | 0.913 → 0.912 | 0.900 / 0.908 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15005 / 2072 | True | 0.881 → 0.894 | 0.802 → 0.823 | 0.905 / 0.913 | False → False |
| sodium@24h | target_specific_change_hurdle | 15000 / 5571 | True | 0.904 → 0.920 | 0.914 → 0.928 | 0.901 / 0.911 | True → True |
| creatinine@48h | target_specific_change_hurdle | 14326 / 5904 | True | 0.883 → 0.926 | 0.856 → 0.920 | 0.901 / 0.920 | False → True |
| glucose@48h | target_specific_change_hurdle | 15001 / 3624 | True | 0.837 → 0.869 | 0.835 → 0.881 | 0.900 / 0.916 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15006 / 1852 | True | 0.907 → 0.920 | 0.918 → 0.927 | 0.903 / 0.919 | True → True |
| map@48h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| potassium@48h | measurement_pure_ridge | 15001 / 5865 | True | 0.908 → 0.911 | 0.917 → 0.919 | 0.902 / 0.906 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
