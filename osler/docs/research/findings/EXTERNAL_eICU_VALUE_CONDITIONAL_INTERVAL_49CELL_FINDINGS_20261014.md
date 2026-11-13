# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `49`; evaluable: `49`
- Point pass: `33`
- Interval pass, fixed → value-conditional: `13` → `24`
- Full pass, fixed → value-conditional: `12` → `21`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 8510 / 1475 | True | 0.831 → 0.863 | 0.812 → 0.872 | 0.901 / 0.902 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15011 / 1063 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | 15002 / 1162 | False | 0.864 → 0.897 | 0.865 → 0.901 | 0.902 / 0.917 | False → False |
| platelets@1h | target_specific_change_hurdle | 282 / 203 | False | 0.842 → 0.825 | 0.794 → 0.820 | 0.900 / 0.903 | False → False |
| urine_output@1h | target_specific_change_hurdle | 15008 / 1507 | False | 0.723 → 0.787 | 0.700 → 0.767 | 0.903 / 0.914 | False → False |
| glucose@3h | observed_treatment_context_ridge | 15004 / 1475 | False | 0.824 → 0.856 | 0.778 → 0.849 | 0.899 / 0.909 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15015 / 1048 | False | 0.915 → 0.929 | 0.907 → 0.923 | 0.904 / 0.923 | False → False |
| hematocrit@3h | measurement_pure_ridge | 3104 / 917 | False | 0.875 → 0.892 | 0.904 → 0.913 | 0.900 / 0.919 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 3439 / 962 | False | 0.860 → 0.865 | 0.852 → 0.857 | 0.900 / 0.912 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15016 / 968 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15010 / 1051 | True | 0.850 → 0.882 | 0.851 → 0.884 | 0.902 / 0.916 | False → True |
| platelets@3h | target_specific_change_hurdle | 1117 / 540 | False | 0.823 → 0.823 | 0.850 → 0.838 | 0.901 / 0.912 | False → False |
| potassium@3h | measurement_pure_ridge | 3411 / 1066 | False | 0.873 → 0.892 | 0.893 → 0.904 | 0.903 / 0.913 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15006 / 941 | True | 0.891 → 0.906 | 0.893 → 0.907 | 0.904 / 0.906 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15007 / 1216 | False | 0.687 → 0.726 | 0.631 → 0.670 | 0.905 / 0.907 | False → False |
| creatinine@6h | target_specific_change_hurdle | 15013 / 3316 | True | 0.732 → 0.833 | 0.758 → 0.837 | 0.899 / 0.927 | False → False |
| glucose@6h | observed_treatment_context_ridge | 13791 / 1072 | False | 0.786 → 0.840 | 0.766 → 0.840 | 0.900 / 0.925 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15000 / 1146 | True | 0.887 → 0.913 | 0.878 → 0.907 | 0.903 / 0.931 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | 3542 / 714 | False | 0.820 → 0.849 | 0.816 → 0.848 | 0.902 / 0.918 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3663 / 721 | True | 0.862 → 0.880 | 0.850 → 0.872 | 0.899 / 0.916 | False → True |
| map@6h | measurement_pure_ridge | 15027 / 1143 | True | 0.843 → 0.871 | 0.842 → 0.867 | 0.904 / 0.912 | False → False |
| potassium@6h | measurement_pure_ridge | 4411 / 909 | True | 0.866 → 0.886 | 0.901 → 0.913 | 0.902 / 0.911 | False → True |
| respiratory_rate@6h | measurement_pure_ridge | 15017 / 1023 | True | 0.861 → 0.891 | 0.876 → 0.903 | 0.902 / 0.919 | False → True |
| urine_output@6h | target_specific_change_hurdle | 15013 / 1155 | False | 0.619 → 0.694 | 0.550 → 0.637 | 0.903 / 0.915 | False → False |
| bicarbonate@12h | target_specific_change_hurdle | 7008 / 1448 | True | 0.796 → 0.823 | 0.829 → 0.846 | 0.900 / 0.917 | False → False |
| chloride@12h | target_specific_change_hurdle | 4707 / 1294 | True | 0.804 → 0.813 | 0.801 → 0.810 | 0.902 / 0.912 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15004 / 2965 | True | 0.759 → 0.855 | 0.766 → 0.873 | 0.900 / 0.930 | False → False |
| glucose@12h | observed_treatment_context_ridge | 15006 / 1167 | False | 0.809 → 0.867 | 0.771 → 0.844 | 0.902 / 0.918 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15018 / 1132 | True | 0.880 → 0.925 | 0.859 → 0.915 | 0.903 / 0.939 | False → True |
| hematocrit@12h | target_specific_change_hurdle | 6787 / 1362 | False | 0.836 → 0.879 | 0.844 → 0.881 | 0.900 / 0.926 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 7263 / 1387 | True | 0.880 → 0.894 | 0.870 → 0.881 | 0.900 / 0.923 | True → True |
| map@12h | measurement_pure_ridge | 15000 / 1143 | True | 0.855 → 0.878 | 0.850 → 0.872 | 0.905 / 0.915 | False → True |
| o2sat@12h | measurement_pure_ridge | 15010 / 964 | True | 0.738 → 0.752 | 0.754 → 0.771 | 0.906 / 0.920 | False → False |
| potassium@12h | measurement_pure_ridge | 7533 / 1445 | True | 0.873 → 0.883 | 0.905 → 0.911 | 0.904 / 0.902 | True → True |
| respiratory_rate@12h | measurement_pure_ridge | 15023 / 999 | True | 0.856 → 0.865 | 0.856 → 0.868 | 0.899 / 0.911 | False → False |
| bicarbonate@24h | target_specific_change_hurdle | 15004 / 1792 | True | 0.871 → 0.884 | 0.878 → 0.887 | 0.902 / 0.906 | True → True |
| bun@24h | target_specific_change_hurdle | 15005 / 2057 | True | 0.793 → 0.888 | 0.801 → 0.890 | 0.900 / 0.916 | False → True |
| creatinine@24h | target_specific_change_hurdle | 15008 / 2047 | True | 0.778 → 0.895 | 0.787 → 0.892 | 0.901 / 0.923 | False → True |
| glucose@24h | measurement_pure_ridge | 15012 / 1084 | True | 0.864 → 0.880 | 0.865 → 0.892 | 0.899 / 0.909 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15006 / 866 | True | 0.896 → 0.943 | 0.906 → 0.945 | 0.904 / 0.944 | True → False |
| map@24h | measurement_pure_ridge | 15012 / 849 | True | 0.897 → 0.895 | 0.900 → 0.899 | 0.902 / 0.911 | True → True |
| potassium@24h | measurement_pure_ridge | 15014 / 1828 | True | 0.887 → 0.888 | 0.884 → 0.886 | 0.900 / 0.908 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15012 / 870 | True | 0.864 → 0.880 | 0.860 → 0.875 | 0.905 / 0.913 | False → True |
| sodium@24h | target_specific_change_hurdle | 15003 / 1963 | True | 0.900 → 0.920 | 0.892 → 0.917 | 0.901 / 0.911 | True → True |
| creatinine@48h | target_specific_change_hurdle | 15008 / 1977 | True | 0.797 → 0.915 | 0.807 → 0.923 | 0.901 / 0.920 | False → True |
| glucose@48h | target_specific_change_hurdle | 15005 / 1039 | True | 0.846 → 0.868 | 0.821 → 0.865 | 0.900 / 0.916 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15012 / 784 | True | 0.922 → 0.938 | 0.921 → 0.933 | 0.903 / 0.919 | True → False |
| map@48h | observed_treatment_context_ridge | 15021 / 812 | False | 0.837 → 0.864 | 0.833 → 0.863 | 0.903 / 0.917 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 1785 | True | 0.859 → 0.861 | 0.861 → 0.862 | 0.902 / 0.906 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
