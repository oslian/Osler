# MIMIC-IV to NWICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient'], interval ['subject_id', 'careunit'], coverage in [0.87, 0.93].

- Catalog cells: `49`; evaluable: `40`
- Point pass: `40`
- Interval pass, fixed → value-conditional: `24` → `28`
- Full pass, fixed → value-conditional: `26` → `30`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 15002 / 5524 | True | 0.872 → 0.888 | 0.785 → 0.837 | 0.901 / 0.900 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15020 / 2161 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| platelets@1h | target_specific_change_hurdle | 2787 / 1814 | True | 0.912 → 0.910 | 0.825 → 0.826 | 0.900 / 0.901 | False → False |
| urine_output@1h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |
| glucose@3h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1980 | True | 0.908 → 0.909 | 0.906 → 0.905 | 0.904 / 0.901 | True → True |
| hematocrit@3h | measurement_pure_ridge | 10458 / 4549 | True | 0.916 → 0.915 | 0.891 → 0.893 | 0.900 / 0.899 | True → True |
| hemoglobin@3h | target_specific_change_hurdle | 10828 / 4727 | True | 0.913 → 0.910 | 0.899 → 0.897 | 0.900 / 0.901 | True → True |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15002 / 2026 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15004 / 2001 | True | 0.914 → 0.923 | 0.905 → 0.910 | 0.902 / 0.900 | True → True |
| platelets@3h | target_specific_change_hurdle | 9051 / 4100 | True | 0.907 → 0.905 | 0.925 → 0.920 | 0.901 / 0.902 | True → True |
| potassium@3h | measurement_pure_ridge | 15014 / 5690 | True | 0.911 → 0.904 | 0.917 → 0.915 | 0.903 / 0.901 | True → True |
| respiratory_rate@3h | measurement_pure_ridge | 15002 / 2117 | True | 0.896 → 0.910 | 0.873 → 0.891 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |
| creatinine@6h | target_specific_change_hurdle | 15000 / 5376 | True | 0.877 → 0.876 | 0.903 → 0.918 | 0.899 / 0.899 | True → True |
| glucose@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@6h | measurement_pure_ridge | 15003 / 1996 | True | 0.890 → 0.887 | 0.904 → 0.897 | 0.903 / 0.900 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| hemoglobin@6h | target_specific_change_hurdle | 14808 / 5986 | True | 0.918 → 0.917 | 0.902 → 0.901 | 0.899 / 0.900 | True → True |
| map@6h | measurement_pure_ridge | 15000 / 2043 | True | 0.897 → 0.911 | 0.909 → 0.923 | 0.904 / 0.900 | True → True |
| potassium@6h | measurement_pure_ridge | 15002 / 5145 | True | 0.895 → 0.899 | 0.775 → 0.838 | 0.902 / 0.897 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15008 / 2164 | True | 0.892 → 0.898 | 0.859 → 0.878 | 0.902 / 0.900 | False → True |
| urine_output@6h | target_specific_change_hurdle | - | not evaluable: target_not_recorded_in_nwicu | | | | |
| bicarbonate@12h | target_specific_change_hurdle | 15001 / 5376 | True | 0.871 → 0.876 | 0.873 → 0.891 | 0.900 / 0.899 | True → True |
| chloride@12h | target_specific_change_hurdle | 15000 / 5308 | True | 0.899 → 0.896 | 0.829 → 0.826 | 0.902 / 0.903 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15010 / 5277 | True | 0.871 → 0.867 | 0.863 → 0.879 | 0.900 / 0.899 | False → False |
| glucose@12h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| heart_rate@12h | measurement_pure_ridge | 15007 / 2122 | True | 0.883 → 0.882 | 0.917 → 0.915 | 0.903 / 0.905 | True → True |
| hematocrit@12h | target_specific_change_hurdle | 11888 / 4656 | True | 0.879 → 0.879 | 0.871 → 0.866 | 0.900 / 0.901 | True → False |
| hemoglobin@12h | measurement_pure_ridge | 12111 / 4753 | True | 0.915 → 0.908 | 0.857 → 0.852 | 0.900 / 0.903 | False → False |
| map@12h | measurement_pure_ridge | 15003 / 2086 | True | 0.902 → 0.903 | 0.891 → 0.898 | 0.905 / 0.901 | True → True |
| o2sat@12h | measurement_pure_ridge | 15004 / 2120 | True | 0.894 → 0.898 | 0.908 → 0.908 | 0.906 / 0.900 | True → True |
| potassium@12h | measurement_pure_ridge | 15001 / 5040 | True | 0.912 → 0.910 | 0.918 → 0.896 | 0.904 / 0.900 | True → True |
| respiratory_rate@12h | measurement_pure_ridge | 15006 / 2206 | True | 0.890 → 0.897 | 0.868 → 0.874 | 0.899 / 0.904 | False → True |
| bicarbonate@24h | target_specific_change_hurdle | 15003 / 6196 | True | 0.894 → 0.899 | 0.884 → 0.887 | 0.902 / 0.901 | True → True |
| bun@24h | target_specific_change_hurdle | 15003 / 6203 | True | 0.882 → 0.877 | 0.832 → 0.834 | 0.900 / 0.898 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15000 / 6178 | True | 0.891 → 0.913 | 0.872 → 0.921 | 0.901 / 0.897 | True → True |
| glucose@24h | measurement_pure_ridge | 15002 / 3612 | True | 0.868 → 0.893 | 0.846 → 0.871 | 0.899 / 0.898 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15005 / 1914 | True | 0.899 → 0.894 | 0.899 → 0.900 | 0.904 / 0.903 | True → True |
| map@24h | measurement_pure_ridge | 15001 / 1984 | True | 0.916 → 0.913 | 0.931 → 0.922 | 0.902 / 0.901 | False → True |
| potassium@24h | measurement_pure_ridge | 15001 / 5921 | True | 0.918 → 0.906 | 0.913 → 0.905 | 0.900 / 0.903 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15005 / 2072 | True | 0.881 → 0.886 | 0.802 → 0.815 | 0.905 / 0.903 | False → False |
| sodium@24h | target_specific_change_hurdle | 15000 / 5571 | True | 0.904 → 0.906 | 0.914 → 0.917 | 0.901 / 0.899 | True → True |
| creatinine@48h | target_specific_change_hurdle | 14326 / 5904 | True | 0.883 → 0.901 | 0.856 → 0.883 | 0.901 / 0.901 | False → True |
| glucose@48h | target_specific_change_hurdle | 15001 / 3624 | True | 0.837 → 0.857 | 0.835 → 0.871 | 0.900 / 0.897 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15006 / 1852 | True | 0.907 → 0.905 | 0.918 → 0.920 | 0.903 / 0.897 | True → True |
| map@48h | observed_treatment_context_ridge | - | not evaluable: treatment_context_not_available_in_nwicu | | | | |
| potassium@48h | measurement_pure_ridge | 15001 / 5865 | True | 0.908 → 0.902 | 0.917 → 0.894 | 0.902 / 0.899 | True → True |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; NWICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
