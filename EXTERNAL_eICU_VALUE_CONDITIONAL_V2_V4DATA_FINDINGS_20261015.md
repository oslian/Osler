# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v4.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `49`; evaluable: `49`
- Point pass: `34`
- Interval pass, fixed → value-conditional: `11` → `17`
- Full pass, fixed → value-conditional: `13` → `17`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 333 / 196 | True | 0.734 → 0.811 | 0.640 → 0.793 | 0.901 / 0.898 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15025 / 1204 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | 15024 / 1255 | False | 0.859 → 0.876 | 0.864 → 0.883 | 0.904 / 0.904 | False → False |
| platelets@1h | target_specific_change_hurdle | 226 / 176 | False | 0.848 → 0.820 | 0.838 → 0.855 | 0.900 / 0.904 | False → False |
| urine_output@1h | target_specific_change_hurdle | 15016 / 1497 | True | 0.879 → 0.865 | 0.896 → 0.871 | 0.900 / 0.902 | True → False |
| glucose@3h | observed_treatment_context_ridge | 1022 / 452 | False | 0.746 → 0.779 | 0.697 → 0.758 | 0.901 / 0.900 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1108 | True | 0.900 → 0.898 | 0.886 → 0.890 | 0.903 / 0.902 | True → True |
| hematocrit@3h | measurement_pure_ridge | 2637 / 861 | False | 0.867 → 0.867 | 0.855 → 0.881 | 0.900 / 0.900 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 2918 / 904 | False | 0.858 → 0.855 | 0.877 → 0.861 | 0.899 / 0.901 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15014 / 1027 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15005 / 1113 | True | 0.854 → 0.871 | 0.859 → 0.876 | 0.900 / 0.899 | False → True |
| platelets@3h | target_specific_change_hurdle | 954 / 486 | False | 0.830 → 0.807 | 0.856 → 0.831 | 0.900 / 0.902 | False → False |
| potassium@3h | measurement_pure_ridge | 2886 / 994 | False | 0.868 → 0.871 | 0.909 → 0.900 | 0.902 / 0.903 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15016 / 1040 | True | 0.894 → 0.907 | 0.883 → 0.904 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15004 / 1255 | True | 0.905 → 0.905 | 0.870 → 0.888 | 0.904 / 0.903 | False → True |
| creatinine@6h | target_specific_change_hurdle | 15009 / 4527 | True | 0.750 → 0.770 | 0.746 → 0.779 | 0.897 / 0.898 | False → False |
| glucose@6h | observed_treatment_context_ridge | 1483 / 630 | False | 0.762 → 0.784 | 0.664 → 0.720 | 0.900 / 0.897 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15006 / 1090 | True | 0.876 → 0.877 | 0.856 → 0.864 | 0.902 / 0.902 | False → False |
| hematocrit@6h | observed_treatment_context_ridge | 3585 / 1017 | False | 0.818 → 0.815 | 0.848 → 0.828 | 0.902 / 0.898 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3950 / 1068 | False | 0.834 → 0.836 | 0.849 → 0.851 | 0.899 / 0.901 | False → False |
| map@6h | measurement_pure_ridge | 15026 / 1127 | True | 0.846 → 0.870 | 0.844 → 0.867 | 0.902 / 0.899 | False → False |
| potassium@6h | measurement_pure_ridge | 3845 / 1125 | True | 0.844 → 0.862 | 0.851 → 0.871 | 0.901 / 0.901 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15004 / 1086 | True | 0.878 → 0.896 | 0.872 → 0.892 | 0.903 / 0.898 | True → True |
| urine_output@6h | target_specific_change_hurdle | 15006 / 1210 | True | 0.914 → 0.917 | 0.899 → 0.905 | 0.903 / 0.904 | True → True |
| bicarbonate@12h | target_specific_change_hurdle | 2945 / 1006 | False | 0.785 → 0.796 | 0.769 → 0.775 | 0.902 / 0.897 | False → False |
| chloride@12h | target_specific_change_hurdle | 1778 / 751 | True | 0.765 → 0.758 | 0.740 → 0.734 | 0.902 / 0.902 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15000 / 4813 | True | 0.747 → 0.788 | 0.716 → 0.766 | 0.900 / 0.900 | False → False |
| glucose@12h | observed_treatment_context_ridge | 1296 / 633 | False | 0.766 → 0.819 | 0.666 → 0.736 | 0.901 / 0.902 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15020 / 1124 | True | 0.871 → 0.877 | 0.844 → 0.857 | 0.905 / 0.902 | False → False |
| hematocrit@12h | target_specific_change_hurdle | 3029 / 937 | False | 0.806 → 0.795 | 0.818 → 0.812 | 0.901 / 0.897 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 3311 / 981 | True | 0.852 → 0.840 | 0.857 → 0.846 | 0.900 / 0.902 | False → False |
| map@12h | measurement_pure_ridge | 15004 / 1166 | True | 0.856 → 0.860 | 0.867 → 0.872 | 0.902 / 0.899 | False → False |
| o2sat@12h | measurement_pure_ridge | 15006 / 1080 | True | 0.839 → 0.840 | 0.839 → 0.843 | 0.905 / 0.903 | False → False |
| potassium@12h | measurement_pure_ridge | 3254 / 1038 | True | 0.864 → 0.870 | 0.871 → 0.898 | 0.903 / 0.901 | False → False |
| respiratory_rate@12h | measurement_pure_ridge | 15007 / 1051 | True | 0.866 → 0.879 | 0.871 → 0.886 | 0.900 / 0.905 | False → True |
| bicarbonate@24h | target_specific_change_hurdle | 15001 / 3977 | True | 0.855 → 0.867 | 0.848 → 0.861 | 0.899 / 0.899 | False → False |
| bun@24h | target_specific_change_hurdle | 15000 / 4917 | True | 0.790 → 0.835 | 0.796 → 0.843 | 0.903 / 0.904 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15000 / 4978 | True | 0.779 → 0.864 | 0.787 → 0.885 | 0.903 / 0.899 | False → False |
| glucose@24h | measurement_pure_ridge | 15007 / 4851 | False | 0.842 → 0.816 | 0.819 → 0.796 | 0.900 / 0.900 | False → False |
| heart_rate@24h | measurement_pure_ridge | 15017 / 925 | True | 0.883 → 0.887 | 0.889 → 0.897 | 0.904 / 0.903 | True → True |
| map@24h | measurement_pure_ridge | 15008 / 893 | True | 0.909 → 0.895 | 0.909 → 0.885 | 0.901 / 0.901 | True → True |
| potassium@24h | measurement_pure_ridge | 15001 / 4266 | True | 0.873 → 0.871 | 0.875 → 0.876 | 0.899 / 0.901 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15017 / 933 | True | 0.891 → 0.891 | 0.908 → 0.903 | 0.905 / 0.901 | True → True |
| sodium@24h | target_specific_change_hurdle | 15002 / 4459 | True | 0.900 → 0.903 | 0.897 → 0.902 | 0.901 / 0.900 | True → True |
| creatinine@48h | target_specific_change_hurdle | 15000 / 4917 | True | 0.802 → 0.874 | 0.801 → 0.881 | 0.901 / 0.903 | False → True |
| glucose@48h | target_specific_change_hurdle | 15003 / 4954 | True | 0.872 → 0.888 | 0.841 → 0.872 | 0.899 / 0.901 | False → True |
| heart_rate@48h | measurement_pure_ridge | 15010 / 910 | True | 0.912 → 0.905 | 0.916 → 0.910 | 0.903 / 0.899 | True → True |
| map@48h | observed_treatment_context_ridge | 15024 / 907 | False | 0.854 → 0.855 | 0.863 → 0.861 | 0.904 / 0.905 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 4180 | True | 0.862 → 0.860 | 0.848 → 0.846 | 0.900 / 0.899 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
