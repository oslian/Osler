# MIMIC-IV to eICU: Fixed vs Value-Conditional Intervals

MIMIC data: `mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet`. Gate axes: point ['patient', 'hospital'], interval ['subject_id', 'hospitalid'], coverage in [0.87, 0.93].

- Catalog cells: `49`; evaluable: `49`
- Point pass: `33`
- Interval pass, fixed → value-conditional: `13` → `18`
- Full pass, fixed → value-conditional: `12` → `16`

| Cell | Source | Rows / patients | Point | Patient cov. fixed → cond. | Group cov. fixed → cond. | MIMIC CV fixed / cond. | Full fixed → cond. |
|---|---|---|---|---|---|---|---|
| glucose@1h | target_specific_change_hurdle | 8510 / 1475 | True | 0.831 → 0.856 | 0.812 → 0.864 | 0.901 / 0.900 | False → False |
| hypoxemia_event@1h | mimiciv_hypoxemia_event_model | 15011 / 1063 | True | event (Brier) | | | True |
| map@1h | observed_treatment_context_ridge | 15002 / 1162 | False | 0.864 → 0.881 | 0.865 → 0.882 | 0.902 / 0.905 | False → False |
| platelets@1h | target_specific_change_hurdle | 282 / 203 | False | 0.842 → 0.814 | 0.794 → 0.828 | 0.900 / 0.901 | False → False |
| urine_output@1h | target_specific_change_hurdle | 15008 / 1507 | False | 0.723 → 0.793 | 0.700 → 0.782 | 0.903 / 0.904 | False → False |
| glucose@3h | observed_treatment_context_ridge | 15004 / 1475 | False | 0.824 → 0.842 | 0.778 → 0.830 | 0.899 / 0.902 | False → False |
| heart_rate@3h | measurement_pure_ridge | 15015 / 1048 | False | 0.915 → 0.916 | 0.907 → 0.911 | 0.904 / 0.901 | False → False |
| hematocrit@3h | measurement_pure_ridge | 3104 / 917 | False | 0.875 → 0.871 | 0.904 → 0.900 | 0.900 / 0.899 | False → False |
| hemoglobin@3h | target_specific_change_hurdle | 3439 / 962 | False | 0.860 → 0.851 | 0.852 → 0.837 | 0.900 / 0.901 | False → False |
| hypoxemia_event@3h | mimiciv_hypoxemia_event_model | 15016 / 968 | True | event (Brier) | | | True |
| map@3h | measurement_pure_ridge | 15010 / 1051 | True | 0.850 → 0.872 | 0.851 → 0.873 | 0.902 / 0.900 | False → True |
| platelets@3h | target_specific_change_hurdle | 1117 / 540 | False | 0.823 → 0.800 | 0.850 → 0.834 | 0.901 / 0.902 | False → False |
| potassium@3h | measurement_pure_ridge | 3411 / 1066 | False | 0.873 → 0.875 | 0.893 → 0.893 | 0.903 / 0.901 | False → False |
| respiratory_rate@3h | measurement_pure_ridge | 15006 / 941 | True | 0.891 → 0.909 | 0.893 → 0.909 | 0.904 / 0.904 | True → True |
| urine_output@3h | target_specific_change_hurdle | 15007 / 1216 | False | 0.687 → 0.789 | 0.631 → 0.749 | 0.905 / 0.904 | False → False |
| creatinine@6h | target_specific_change_hurdle | 15013 / 3316 | True | 0.732 → 0.759 | 0.758 → 0.776 | 0.899 / 0.899 | False → False |
| glucose@6h | observed_treatment_context_ridge | 13791 / 1072 | False | 0.786 → 0.802 | 0.766 → 0.807 | 0.900 / 0.899 | False → False |
| heart_rate@6h | measurement_pure_ridge | 15000 / 1146 | True | 0.887 → 0.887 | 0.878 → 0.884 | 0.903 / 0.900 | True → True |
| hematocrit@6h | observed_treatment_context_ridge | 3542 / 714 | False | 0.820 → 0.816 | 0.816 → 0.812 | 0.902 / 0.898 | False → False |
| hemoglobin@6h | target_specific_change_hurdle | 3663 / 721 | True | 0.862 → 0.860 | 0.850 → 0.842 | 0.899 / 0.900 | False → False |
| map@6h | measurement_pure_ridge | 15027 / 1143 | True | 0.843 → 0.864 | 0.842 → 0.860 | 0.904 / 0.900 | False → False |
| potassium@6h | measurement_pure_ridge | 4411 / 909 | True | 0.866 → 0.877 | 0.901 → 0.899 | 0.902 / 0.897 | False → True |
| respiratory_rate@6h | measurement_pure_ridge | 15017 / 1023 | True | 0.861 → 0.879 | 0.876 → 0.894 | 0.902 / 0.900 | False → True |
| urine_output@6h | target_specific_change_hurdle | 15013 / 1155 | False | 0.619 → 0.687 | 0.550 → 0.627 | 0.903 / 0.903 | False → False |
| bicarbonate@12h | target_specific_change_hurdle | 7008 / 1448 | True | 0.796 → 0.803 | 0.829 → 0.828 | 0.900 / 0.899 | False → False |
| chloride@12h | target_specific_change_hurdle | 4707 / 1294 | True | 0.804 → 0.797 | 0.801 → 0.789 | 0.902 / 0.903 | False → False |
| creatinine@12h | target_specific_change_hurdle | 15004 / 2965 | True | 0.759 → 0.779 | 0.766 → 0.802 | 0.900 / 0.899 | False → False |
| glucose@12h | observed_treatment_context_ridge | 15006 / 1167 | False | 0.809 → 0.841 | 0.771 → 0.823 | 0.902 / 0.901 | False → False |
| heart_rate@12h | measurement_pure_ridge | 15018 / 1132 | True | 0.880 → 0.885 | 0.859 → 0.865 | 0.903 / 0.905 | False → False |
| hematocrit@12h | target_specific_change_hurdle | 6787 / 1362 | False | 0.836 → 0.834 | 0.844 → 0.840 | 0.900 / 0.901 | False → False |
| hemoglobin@12h | measurement_pure_ridge | 7263 / 1387 | True | 0.880 → 0.867 | 0.870 → 0.853 | 0.900 / 0.903 | True → False |
| map@12h | measurement_pure_ridge | 15000 / 1143 | True | 0.855 → 0.860 | 0.850 → 0.857 | 0.905 / 0.901 | False → False |
| o2sat@12h | measurement_pure_ridge | 15010 / 964 | True | 0.738 → 0.743 | 0.754 → 0.762 | 0.906 / 0.900 | False → False |
| potassium@12h | measurement_pure_ridge | 7533 / 1445 | True | 0.873 → 0.882 | 0.905 → 0.909 | 0.904 / 0.900 | True → True |
| respiratory_rate@12h | measurement_pure_ridge | 15023 / 999 | True | 0.856 → 0.866 | 0.856 → 0.866 | 0.899 / 0.904 | False → False |
| bicarbonate@24h | target_specific_change_hurdle | 15004 / 1792 | True | 0.871 → 0.878 | 0.878 → 0.885 | 0.902 / 0.901 | True → True |
| bun@24h | target_specific_change_hurdle | 15005 / 2057 | True | 0.793 → 0.845 | 0.801 → 0.858 | 0.900 / 0.898 | False → False |
| creatinine@24h | target_specific_change_hurdle | 15008 / 2047 | True | 0.778 → 0.859 | 0.787 → 0.857 | 0.901 / 0.897 | False → False |
| glucose@24h | measurement_pure_ridge | 15012 / 1084 | True | 0.864 → 0.871 | 0.865 → 0.883 | 0.899 / 0.898 | False → True |
| heart_rate@24h | measurement_pure_ridge | 15006 / 866 | True | 0.896 → 0.905 | 0.906 → 0.908 | 0.904 / 0.903 | True → True |
| map@24h | measurement_pure_ridge | 15012 / 849 | True | 0.897 → 0.878 | 0.900 → 0.882 | 0.902 / 0.901 | True → True |
| potassium@24h | measurement_pure_ridge | 15014 / 1828 | True | 0.887 → 0.880 | 0.884 → 0.884 | 0.900 / 0.903 | True → True |
| respiratory_rate@24h | measurement_pure_ridge | 15012 / 870 | True | 0.864 → 0.867 | 0.860 → 0.858 | 0.905 / 0.903 | False → False |
| sodium@24h | target_specific_change_hurdle | 15003 / 1963 | True | 0.900 → 0.904 | 0.892 → 0.898 | 0.901 / 0.899 | True → True |
| creatinine@48h | target_specific_change_hurdle | 15008 / 1977 | True | 0.797 → 0.877 | 0.807 → 0.887 | 0.901 / 0.901 | False → True |
| glucose@48h | target_specific_change_hurdle | 15005 / 1039 | True | 0.846 → 0.851 | 0.821 → 0.848 | 0.900 / 0.897 | False → False |
| heart_rate@48h | measurement_pure_ridge | 15012 / 784 | True | 0.922 → 0.922 | 0.921 → 0.919 | 0.903 / 0.897 | True → True |
| map@48h | observed_treatment_context_ridge | 15021 / 812 | False | 0.837 → 0.841 | 0.833 → 0.839 | 0.903 / 0.907 | False → False |
| potassium@48h | measurement_pure_ridge | 15002 / 1785 | True | 0.859 → 0.854 | 0.861 → 0.853 | 0.902 / 0.899 | False → False |

## Boundary

All fitting, selection and calibration use MIMIC-IV only; eICU labels are read once for scoring.
No result here changes the frozen runtime registry.
`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
