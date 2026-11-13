# Module-Conditional Intervals (v3) on eICU — non-independent re-check

- Evaluable continuous cells: `47`
- Interval pass v2 → v3: `18` → `20`
- Full pass (point gate from the v2 run) v2 → v3: `14` → `15`
- v2 reproduces the v2 audit in every cell: `True`

| Cell | Module | Members | MIMIC held-out members, v2 cov. | MIMIC members, v3 CV | Ext. patient cov. v2 → v3 | Ext. group cov. v2 → v3 | Full v2 → v3 |
|---|---|---:|---:|---:|---|---|---|
| glucose@1h | endocrine_stress | 100% | 0.876 | 0.900 | 0.856 → 0.881 | 0.864 → 0.895 | False → True |
| map@1h | cardiovascular_instability | 100% | 0.897 | 0.900 | 0.881 → 0.891 | 0.882 → 0.891 | False → False |
| platelets@1h | coagulopathy_heme | 100% | - | 0.901 | 0.814 → 0.820 | 0.828 → 0.824 | False → False |
| urine_output@1h | aki | 100% | 0.894 | 0.900 | 0.793 → 0.806 | 0.782 → 0.798 | False → False |
| glucose@3h | endocrine_stress | 100% | 0.867 | 0.900 | 0.842 → 0.877 | 0.830 → 0.864 | False → False |
| heart_rate@3h | cardiovascular_instability | 100% | 0.917 | 0.900 | 0.916 → 0.903 | 0.911 → 0.896 | False → False |
| hematocrit@3h | coagulopathy_heme | 100% | 0.898 | 0.900 | 0.871 → 0.873 | 0.900 → 0.881 | False → False |
| hemoglobin@3h | coagulopathy_heme | 100% | 0.900 | 0.900 | 0.851 → 0.852 | 0.837 → 0.842 | False → False |
| map@3h | cardiovascular_instability | 100% | 0.903 | 0.901 | 0.872 → 0.864 | 0.873 → 0.865 | True → False |
| platelets@3h | coagulopathy_heme | 100% | 0.899 | 0.901 | 0.800 → 0.807 | 0.834 → 0.830 | False → False |
| potassium@3h | electrolyte_acid_base | 100% | 0.889 | 0.900 | 0.875 → 0.892 | 0.893 → 0.908 | False → False |
| respiratory_rate@3h | respiratory | 100% | 0.898 | 0.901 | 0.909 → 0.913 | 0.909 → 0.911 | True → True |
| urine_output@3h | aki | 100% | 0.920 | 0.901 | 0.789 → 0.750 | 0.749 → 0.709 | False → False |
| creatinine@6h | aki | 100% | 0.835 | 0.899 | 0.759 → 0.821 | 0.776 → 0.832 | False → False |
| glucose@6h | endocrine_stress | 100% | 0.833 | 0.899 | 0.802 → 0.876 | 0.807 → 0.872 | False → False |
| heart_rate@6h | cardiovascular_instability | 100% | 0.889 | 0.901 | 0.887 → 0.895 | 0.884 → 0.890 | True → True |
| hematocrit@6h | coagulopathy_heme | 100% | 0.892 | 0.900 | 0.816 → 0.820 | 0.812 → 0.818 | False → False |
| hemoglobin@6h | coagulopathy_heme | 100% | 0.916 | 0.899 | 0.860 → 0.842 | 0.842 → 0.826 | False → False |
| map@6h | cardiovascular_instability | 100% | 0.891 | 0.901 | 0.864 → 0.874 | 0.860 → 0.869 | False → False |
| potassium@6h | electrolyte_acid_base | 100% | 0.877 | 0.901 | 0.877 → 0.897 | 0.899 → 0.923 | True → True |
| respiratory_rate@6h | respiratory | 100% | 0.891 | 0.900 | 0.879 → 0.890 | 0.894 → 0.902 | True → True |
| urine_output@6h | aki | 100% | 0.928 | 0.901 | 0.687 → 0.638 | 0.627 → 0.574 | False → False |
| bicarbonate@12h | electrolyte_acid_base | 100% | 0.887 | 0.899 | 0.803 → 0.819 | 0.828 → 0.845 | False → False |
| chloride@12h | electrolyte_acid_base | 100% | 0.873 | 0.900 | 0.797 → 0.829 | 0.789 → 0.834 | False → False |
| creatinine@12h | aki | 100% | 0.821 | 0.901 | 0.779 → 0.861 | 0.802 → 0.870 | False → False |
| glucose@12h | endocrine_stress | 100% | 0.880 | 0.900 | 0.841 → 0.869 | 0.823 → 0.859 | False → False |
| heart_rate@12h | cardiovascular_instability | 100% | 0.878 | 0.902 | 0.885 → 0.899 | 0.865 → 0.882 | False → True |
| hematocrit@12h | coagulopathy_heme | 100% | 0.876 | 0.900 | 0.834 → 0.860 | 0.840 → 0.862 | False → False |
| hemoglobin@12h | coagulopathy_heme | 100% | 0.888 | 0.900 | 0.867 → 0.875 | 0.853 → 0.859 | False → False |
| map@12h | cardiovascular_instability | 100% | 0.891 | 0.900 | 0.860 → 0.872 | 0.857 → 0.869 | False → False |
| o2sat@12h | respiratory | 100% | 0.845 | 0.901 | 0.743 → 0.800 | 0.762 → 0.823 | False → False |
| potassium@12h | electrolyte_acid_base | 100% | 0.889 | 0.901 | 0.882 → 0.893 | 0.909 → 0.924 | True → True |
| respiratory_rate@12h | respiratory | 100% | 0.867 | 0.901 | 0.866 → 0.892 | 0.866 → 0.891 | False → True |
| bicarbonate@24h | electrolyte_acid_base | 3% | 0.901 | 0.900 | 0.878 → 0.863 | 0.885 → 0.869 | True → False |
| bun@24h | aki | 100% | 0.877 | 0.899 | 0.845 → 0.880 | 0.858 → 0.892 | False → True |
| creatinine@24h | aki | 100% | 0.858 | 0.900 | 0.859 → 0.889 | 0.857 → 0.893 | False → True |
| glucose@24h | endocrine_stress | 2% | 0.874 | 0.901 | 0.871 → 0.805 | 0.883 → 0.807 | True → False |
| heart_rate@24h | cardiovascular_instability | 2% | 0.892 | 0.900 | 0.905 → 0.871 | 0.908 → 0.882 | True → True |
| map@24h | cardiovascular_instability | 2% | 0.911 | 0.901 | 0.878 → 0.902 | 0.882 → 0.903 | True → True |
| potassium@24h | electrolyte_acid_base | 3% | 0.881 | 0.901 | 0.880 → 0.870 | 0.884 → 0.868 | True → False |
| respiratory_rate@24h | respiratory | 32% | 0.857 | 0.901 | 0.867 → 0.877 | 0.858 → 0.873 | False → True |
| sodium@24h | electrolyte_acid_base | 3% | 0.891 | 0.901 | 0.904 → 0.865 | 0.898 → 0.859 | True → False |
| creatinine@48h | aki | 100% | 0.842 | 0.900 | 0.877 → 0.923 | 0.887 → 0.916 | True → True |
| glucose@48h | endocrine_stress | 4% | 0.848 | 0.900 | 0.851 → 0.777 | 0.848 → 0.758 | False → False |
| heart_rate@48h | cardiovascular_instability | 3% | 0.900 | 0.899 | 0.922 → 0.915 | 0.919 → 0.916 | True → True |
| map@48h | cardiovascular_instability | 2% | 0.900 | 0.900 | 0.841 → 0.821 | 0.839 → 0.818 | False → False |
| potassium@48h | electrolyte_acid_base | 3% | 0.885 | 0.901 | 0.854 → 0.829 | 0.853 → 0.826 | False → False |

Non-independent: this database was already used. No promotion. `causal_claim_allowed: false`.
