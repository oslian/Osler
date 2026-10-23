# Module-Conditional Intervals (v3) on NWICU — non-independent re-check

- Evaluable continuous cells: `38`
- Interval pass v2 → v3: `28` → `24`
- Full pass (point gate from the v2 run) v2 → v3: `28` → `24`
- v2 reproduces the v2 audit in every cell: `True`

| Cell | Module | Members | MIMIC held-out members, v2 cov. | MIMIC members, v3 CV | Ext. patient cov. v2 → v3 | Ext. group cov. v2 → v3 | Full v2 → v3 |
|---|---|---:|---:|---:|---|---|---|
| glucose@1h | endocrine_stress | 31% | 0.876 | 0.900 | 0.888 → 0.878 | 0.837 → 0.830 | False → False |
| platelets@1h | coagulopathy_heme | 29% | - | 0.901 | 0.910 → 0.917 | 0.826 → 0.833 | False → False |
| heart_rate@3h | cardiovascular_instability | 20% | 0.917 | 0.900 | 0.909 → 0.906 | 0.905 → 0.904 | True → True |
| hematocrit@3h | coagulopathy_heme | 34% | 0.898 | 0.900 | 0.915 → 0.913 | 0.893 → 0.890 | True → True |
| hemoglobin@3h | coagulopathy_heme | 34% | 0.900 | 0.900 | 0.910 → 0.905 | 0.897 → 0.894 | True → True |
| map@3h | cardiovascular_instability | 19% | 0.903 | 0.901 | 0.923 → 0.918 | 0.910 → 0.907 | True → True |
| platelets@3h | coagulopathy_heme | 28% | 0.899 | 0.901 | 0.905 → 0.900 | 0.920 → 0.916 | True → True |
| potassium@3h | electrolyte_acid_base | 27% | 0.889 | 0.900 | 0.904 → 0.901 | 0.915 → 0.912 | True → True |
| respiratory_rate@3h | respiratory | 7% | 0.898 | 0.901 | 0.910 → 0.906 | 0.891 → 0.888 | True → True |
| creatinine@6h | aki | 54% | 0.835 | 0.899 | 0.876 → 0.823 | 0.918 → 0.796 | True → False |
| heart_rate@6h | cardiovascular_instability | 18% | 0.889 | 0.901 | 0.887 → 0.895 | 0.897 → 0.906 | True → True |
| hemoglobin@6h | coagulopathy_heme | 33% | 0.916 | 0.899 | 0.917 → 0.908 | 0.901 → 0.884 | True → True |
| map@6h | cardiovascular_instability | 19% | 0.891 | 0.901 | 0.911 → 0.902 | 0.923 → 0.917 | True → True |
| potassium@6h | electrolyte_acid_base | 25% | 0.877 | 0.901 | 0.899 → 0.884 | 0.838 → 0.829 | False → False |
| respiratory_rate@6h | respiratory | 7% | 0.891 | 0.900 | 0.898 → 0.898 | 0.878 → 0.878 | True → True |
| bicarbonate@12h | electrolyte_acid_base | 26% | 0.887 | 0.899 | 0.876 → 0.869 | 0.891 → 0.900 | True → False |
| chloride@12h | electrolyte_acid_base | 26% | 0.873 | 0.900 | 0.896 → 0.906 | 0.826 → 0.898 | False → True |
| creatinine@12h | aki | 58% | 0.821 | 0.901 | 0.867 → 0.907 | 0.879 → 0.903 | False → True |
| heart_rate@12h | cardiovascular_instability | 22% | 0.878 | 0.902 | 0.882 → 0.877 | 0.915 → 0.911 | True → True |
| hematocrit@12h | coagulopathy_heme | 35% | 0.876 | 0.900 | 0.879 → 0.889 | 0.866 → 0.891 | False → True |
| hemoglobin@12h | coagulopathy_heme | 35% | 0.888 | 0.900 | 0.908 → 0.908 | 0.852 → 0.851 | False → False |
| map@12h | cardiovascular_instability | 22% | 0.891 | 0.900 | 0.903 → 0.898 | 0.898 → 0.887 | True → True |
| o2sat@12h | respiratory | 7% | 0.845 | 0.901 | 0.898 → 0.901 | 0.908 → 0.913 | True → True |
| potassium@12h | electrolyte_acid_base | 25% | 0.889 | 0.901 | 0.910 → 0.897 | 0.896 → 0.887 | True → True |
| respiratory_rate@12h | respiratory | 8% | 0.867 | 0.901 | 0.897 → 0.898 | 0.874 → 0.876 | True → True |
| bicarbonate@24h | electrolyte_acid_base | 23% | 0.901 | 0.900 | 0.899 → 0.888 | 0.887 → 0.880 | True → True |
| bun@24h | aki | 54% | 0.877 | 0.899 | 0.877 → 0.855 | 0.834 → 0.832 | False → False |
| creatinine@24h | aki | 54% | 0.858 | 0.900 | 0.913 → 0.834 | 0.921 → 0.848 | True → False |
| glucose@24h | endocrine_stress | 23% | 0.874 | 0.901 | 0.893 → 0.851 | 0.871 → 0.837 | True → False |
| heart_rate@24h | cardiovascular_instability | 22% | 0.892 | 0.900 | 0.894 → 0.869 | 0.900 → 0.876 | True → False |
| map@24h | cardiovascular_instability | 21% | 0.911 | 0.901 | 0.913 → 0.918 | 0.922 → 0.932 | True → False |
| potassium@24h | electrolyte_acid_base | 23% | 0.881 | 0.901 | 0.906 → 0.900 | 0.905 → 0.900 | True → True |
| respiratory_rate@24h | respiratory | 8% | 0.857 | 0.901 | 0.886 → 0.891 | 0.815 → 0.825 | False → False |
| sodium@24h | electrolyte_acid_base | 24% | 0.891 | 0.901 | 0.906 → 0.885 | 0.917 → 0.900 | True → True |
| creatinine@48h | aki | 61% | 0.842 | 0.900 | 0.901 → 0.863 | 0.883 → 0.880 | True → False |
| glucose@48h | endocrine_stress | 24% | 0.848 | 0.900 | 0.857 → 0.813 | 0.871 → 0.824 | False → False |
| heart_rate@48h | cardiovascular_instability | 21% | 0.900 | 0.899 | 0.905 → 0.907 | 0.920 → 0.921 | True → True |
| potassium@48h | electrolyte_acid_base | 24% | 0.885 | 0.901 | 0.902 → 0.889 | 0.894 → 0.886 | True → True |

Non-independent: this database was already used. No promotion. `causal_claim_allowed: false`.
