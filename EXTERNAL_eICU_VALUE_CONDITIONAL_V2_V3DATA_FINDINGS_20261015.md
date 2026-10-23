# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v3.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `49`; evaluable: `49`
- Point pass: `34`
- Interval pass, fixed → value-conditional: `12` → `18`
- Full pass, fixed → value-conditional: `13` → `18`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 333 / 196 | False | 0.657 → 0.769 | 0.571 → 0.760 | 0.902 / 0.899 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15025 / 1204 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | 15024 / 1255 | False | 0.858 → 0.879 | 0.864 → 0.886 | 0.904 / 0.903 | False → False |
| platelets@1h | target_specific_change_hurdle | 226 / 176 | False | 0.854 → 0.820 | 0.839 → 0.855 | 0.900 / 0.902 | False → False |
| urine_output@1h | target_specific_change_hurdle | 15016 / 1497 | True | 0.885 → 0.863 | 0.906 → 0.870 | 0.905 / 0.903 | True → False |
| glucose@3h | observed_treatment_context_ridge | 1022 / 452 | False | 0.728 → 0.785 | 0.681 → 0.754 | 0.900 / 0.902 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15006 / 1108 | True | 0.900 → 0.897 | 0.886 → 0.888 | 0.903 / 0.900 | True → True |
| hematocrit@3h | measurement_pure_ridge | 2637 / 861 | False | 0.864 → 0.866 | 0.852 → 0.881 | 0.900 / 0.900 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 2918 / 904 | False | 0.858 → 0.858 | 0.875 → 0.863 | 0.901 / 0.902 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15014 / 1027 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15005 / 1113 | True | 0.853 → 0.874 | 0.859 → 0.879 | 0.901 / 0.898 | False → True |
| platelets@3h | target_specific_change_hurdle | 954 / 486 | False | 0.827 → 0.808 | 0.850 → 0.829 | 0.901 / 0.901 | False → False |
| potassium@3h | measurement_pure_ridge | 2886 / 994 | False | 0.871 → 0.877 | 0.905 → 0.905 | 0.899 / 0.900 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15016 / 1040 | True | 0.894 → 0.907 | 0.890 → 0.903 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15004 / 1255 | True | 0.903 → 0.906 | 0.868 → 0.886 | 0.903 / 0.903 | False → True |
| creatinine@6h | target_specific_change_hurdle | 15009 / 4527 | True | 0.745 → 0.766 | 0.742 → 0.776 | 0.898 / 0.897 | False → False |
| glucose@6h | observed_treatment_context_ridge | 1483 / 630 | False | 0.699 → 0.760 | 0.584 → 0.693 | 0.902 / 0.897 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15006 / 1090 | True | 0.875 → 0.875 | 0.855 → 0.861 | 0.903 / 0.902 | False → False |
| hematocrit@6h | observed_treatment_context_ridge | 3585 / 1017 | False | 0.821 → 0.814 | 0.849 → 0.828 | 0.902 / 0.899 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3950 / 1068 | False | 0.840 → 0.837 | 0.850 → 0.848 | 0.901 / 0.901 | False → False |
| map@6h | measurement_pure_ridge | 15026 / 1127 | True | 0.849 → 0.870 | 0.847 → 0.867 | 0.902 / 0.903 | False → False |
| potassium@6h | measurement_pure_ridge | 3845 / 1125 | True | 0.844 → 0.861 | 0.852 → 0.871 | 0.902 / 0.897 | False → False |
| respiratory_rate@6h | measurement_pure_ridge | 15004 / 1086 | True | 0.879 → 0.893 | 0.872 → 0.890 | 0.902 / 0.899 | True → True |
| urine_output@6h | target_specific_change_hurdle | 15006 / 1210 | True | 0.914 → 0.918 | 0.899 → 0.905 | 0.903 / 0.904 | True → True |
| bicarbonate@12h | target_specific_change_hurdle | 2945 / 1006 | False | 0.783 → 0.799 | 0.766 → 0.776 | 0.901 / 0.899 | False → False |
| chloride@12h | target_specific_change_hurdle | 1778 / 751 | True | 0.758 → 0.760 | 0.740 → 0.733 | 0.902 / 0.901 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15000 / 4813 | True | 0.749 → 0.788 | 0.723 → 0.767 | 0.902 / 0.899 | False → False |
| glucose@12h | observed_treatment_context_ridge | 1296 / 633 | False | 0.759 → 0.801 | 0.627 → 0.718 | 0.901 / 0.900 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15020 / 1124 | True | 0.870 → 0.872 | 0.843 → 0.851 | 0.906 / 0.903 | False → False |
| hematocrit@12h | target_specific_change_hurdle | 3029 / 937 | False | 0.809 → 0.802 | 0.818 → 0.814 | 0.899 / 0.899 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 3311 / 981 | True | 0.855 → 0.838 | 0.859 → 0.845 | 0.900 / 0.903 | False → False |
| map@12h | measurement_pure_ridge | 15004 / 1166 | True | 0.857 → 0.859 | 0.867 → 0.871 | 0.906 / 0.901 | False → False |
| o2sat@12h | measurement_pure_ridge | 15006 / 1080 | True | 0.837 → 0.841 | 0.839 → 0.846 | 0.904 / 0.899 | False → False |
| potassium@12h | measurement_pure_ridge | 3254 / 1038 | True | 0.862 → 0.870 | 0.870 → 0.897 | 0.903 / 0.899 | False → True |
| respiratory_rate@12h | measurement_pure_ridge | 15007 / 1051 | True | 0.867 → 0.877 | 0.872 → 0.885 | 0.899 / 0.904 | False → True |
| bicarbonate@24h | target_specific_change_hurdle | 15001 / 3977 | True | 0.856 → 0.865 | 0.850 → 0.864 | 0.902 / 0.900 | False → False |
| bun@24h | target_specific_change_hurdle | 15000 / 4917 | True | 0.790 → 0.834 | 0.797 → 0.841 | 0.901 / 0.900 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15000 / 4978 | True | 0.781 → 0.862 | 0.788 → 0.884 | 0.903 / 0.899 | False → False |
| glucose@24h | measurement_pure_ridge | 15007 / 4851 | True | 0.875 → 0.887 | 0.849 → 0.881 | 0.899 / 0.899 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15017 / 925 | True | 0.883 → 0.890 | 0.891 → 0.896 | 0.904 / 0.904 | True → True |
| map@24h | measurement_pure_ridge | 15008 / 893 | True | 0.909 → 0.897 | 0.909 → 0.889 | 0.900 / 0.900 | True → True |
| potassium@24h | measurement_pure_ridge | 15001 / 4266 | True | 0.876 → 0.874 | 0.876 → 0.878 | 0.900 / 0.900 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15017 / 933 | True | 0.891 → 0.890 | 0.908 → 0.901 | 0.906 / 0.901 | True → True |
| sodium@24h | target_specific_change_hurdle | 15002 / 4459 | True | 0.901 → 0.906 | 0.898 → 0.906 | 0.900 / 0.900 | True → True |
| creatinine@48h | target_specific_change_hurdle | 15000 / 4917 | True | 0.799 → 0.874 | 0.798 → 0.880 | 0.900 / 0.901 | False → True |
| glucose@48h | target_specific_change_hurdle | 15003 / 4954 | True | 0.860 → 0.874 | 0.828 → 0.857 | 0.899 / 0.898 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15010 / 910 | True | 0.910 → 0.910 | 0.915 → 0.913 | 0.902 / 0.896 | True → True |
| map@48h | observed_treatment_context_ridge | 15024 / 907 | False | 0.846 → 0.852 | 0.859 → 0.859 | 0.904 / 0.905 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 4180 | True | 0.863 → 0.859 | 0.850 → 0.844 | 0.901 / 0.899 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
