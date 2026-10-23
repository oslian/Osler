# Activity-Conditional Intervals (v4): MIMIC held-out check and eICU re-check

- Cells: `47`
- Rule 1 (MIMIC held-out active and inactive coverage in [0.89, 0.91]): `25` of `46` cells → NOT met (needs ≥ 90%)
- eICU full pass v2 → v4: `14` → `12` (non-independent) → rule 2 NOT met
- v2 reproduces the v2 audit in every cell: `True`

| Cell | Module | MIMIC active share | MIMIC hold-out active v2 → v4 | inactive v2 → v4 | Ext. active share | Ext. patient cov. v2 → v4 | Full v2 → v4 |
|---|---|---:|---|---|---:|---|---|
| glucose@1h | endocrine_stress | 48% | 0.886 → 0.899 | 0.915 → 0.888 | 29% | 0.856 → 0.841 | False → False |
| map@1h | cardiovascular_instability | 43% | 0.886 → 0.890 | 0.901 → 0.892 | 22% | 0.881 → 0.877 | False → False |
| platelets@1h | coagulopathy_heme | nan% | - → - | - → - | 82% | 0.814 → 0.808 | False → False |
| urine_output@1h | aki | 31% | 0.839 → 0.905 | 0.893 → 0.899 | 39% | 0.793 → 0.828 | False → False |
| glucose@3h | endocrine_stress | 49% | 0.865 → 0.889 | 0.924 → 0.903 | 28% | 0.842 → 0.838 | False → False |
| heart_rate@3h | cardiovascular_instability | 44% | 0.913 → 0.915 | 0.911 → 0.893 | 21% | 0.916 → 0.907 | False → False |
| hematocrit@3h | coagulopathy_heme | 79% | 0.890 → 0.900 | 0.920 → 0.906 | 67% | 0.871 → 0.874 | False → False |
| hemoglobin@3h | coagulopathy_heme | 87% | 0.905 → 0.909 | 0.914 → 0.892 | 66% | 0.851 → 0.847 | False → False |
| map@3h | cardiovascular_instability | 43% | 0.898 → 0.913 | 0.907 → 0.900 | 22% | 0.872 → 0.867 | True → False |
| platelets@3h | coagulopathy_heme | 89% | 0.887 → 0.886 | 0.925 → 0.897 | 85% | 0.800 → 0.798 | False → False |
| potassium@3h | electrolyte_acid_base | 68% | 0.896 → 0.908 | 0.912 → 0.898 | 73% | 0.875 → 0.897 | False → False |
| respiratory_rate@3h | respiratory | 52% | 0.908 → 0.906 | 0.911 → 0.898 | 68% | 0.909 → 0.901 | True → True |
| urine_output@3h | aki | 27% | 0.851 → 0.897 | 0.926 → 0.907 | 36% | 0.789 → 0.776 | False → False |
| creatinine@6h | aki | 55% | 0.839 → 0.901 | 0.935 → 0.905 | 73% | 0.759 → 0.806 | False → False |
| glucose@6h | endocrine_stress | 48% | 0.847 → 0.876 | 0.888 → 0.899 | 27% | 0.802 → 0.819 | False → False |
| heart_rate@6h | cardiovascular_instability | 42% | 0.880 → 0.884 | 0.902 → 0.903 | 21% | 0.887 → 0.891 | True → True |
| hematocrit@6h | coagulopathy_heme | 79% | 0.877 → 0.886 | 0.906 → 0.894 | 68% | 0.816 → 0.819 | False → False |
| hemoglobin@6h | coagulopathy_heme | 86% | 0.908 → 0.903 | 0.932 → 0.893 | 67% | 0.860 → 0.845 | False → False |
| map@6h | cardiovascular_instability | 41% | 0.892 → 0.894 | 0.897 → 0.901 | 22% | 0.864 → 0.870 | False → False |
| potassium@6h | electrolyte_acid_base | 68% | 0.887 → 0.894 | 0.902 → 0.891 | 69% | 0.877 → 0.885 | True → True |
| respiratory_rate@6h | respiratory | 52% | 0.899 → 0.896 | 0.919 → 0.903 | 45% | 0.879 → 0.882 | True → True |
| urine_output@6h | aki | 29% | 0.890 → 0.901 | 0.921 → 0.916 | 35% | 0.687 → 0.680 | False → False |
| bicarbonate@12h | electrolyte_acid_base | 75% | 0.889 → 0.899 | 0.938 → 0.914 | 71% | 0.803 → 0.798 | False → False |
| chloride@12h | electrolyte_acid_base | 74% | 0.878 → 0.899 | 0.908 → 0.903 | 85% | 0.797 → 0.817 | False → False |
| creatinine@12h | aki | 52% | 0.842 → 0.906 | 0.915 → 0.890 | 72% | 0.779 → 0.820 | False → False |
| glucose@12h | endocrine_stress | 48% | 0.893 → 0.901 | 0.915 → 0.898 | 28% | 0.841 → 0.829 | False → False |
| heart_rate@12h | cardiovascular_instability | 45% | 0.880 → 0.898 | 0.894 → 0.894 | 22% | 0.885 → 0.887 | False → True |
| hematocrit@12h | coagulopathy_heme | 81% | 0.860 → 0.881 | 0.894 → 0.895 | 69% | 0.834 → 0.853 | False → False |
| hemoglobin@12h | coagulopathy_heme | 87% | 0.890 → 0.894 | 0.913 → 0.907 | 69% | 0.867 → 0.867 | False → False |
| map@12h | cardiovascular_instability | 45% | 0.914 → 0.913 | 0.905 → 0.897 | 22% | 0.860 → 0.854 | False → False |
| o2sat@12h | respiratory | 56% | 0.903 → 0.898 | 0.913 → 0.924 | 70% | 0.743 → 0.743 | False → False |
| potassium@12h | electrolyte_acid_base | 67% | 0.902 → 0.901 | 0.924 → 0.904 | 71% | 0.882 → 0.879 | True → True |
| respiratory_rate@12h | respiratory | 59% | 0.886 → 0.884 | 0.880 → 0.884 | 68% | 0.866 → 0.866 | False → False |
| bicarbonate@24h | electrolyte_acid_base | 71% | 0.897 → 0.890 | 0.943 → 0.913 | 60% | 0.878 → 0.860 | True → False |
| bun@24h | aki | 49% | 0.877 → 0.902 | 0.935 → 0.905 | 68% | 0.845 → 0.852 | False → False |
| creatinine@24h | aki | 49% | 0.884 → 0.902 | 0.926 → 0.894 | 69% | 0.859 → 0.862 | False → False |
| glucose@24h | endocrine_stress | 48% | 0.909 → 0.906 | 0.920 → 0.907 | 24% | 0.871 → 0.863 | True → False |
| heart_rate@24h | cardiovascular_instability | 44% | 0.898 → 0.898 | 0.887 → 0.892 | 42% | 0.905 → 0.897 | True → True |
| map@24h | cardiovascular_instability | 44% | 0.914 → 0.914 | 0.912 → 0.889 | 42% | 0.878 → 0.862 | True → False |
| potassium@24h | electrolyte_acid_base | 68% | 0.897 → 0.917 | 0.911 → 0.904 | 60% | 0.880 → 0.893 | True → True |
| respiratory_rate@24h | respiratory | 62% | 0.874 → 0.894 | 0.885 → 0.891 | 33% | 0.867 → 0.880 | False → True |
| sodium@24h | electrolyte_acid_base | 70% | 0.892 → 0.899 | 0.922 → 0.891 | 65% | 0.904 → 0.896 | True → True |
| creatinine@48h | aki | 53% | 0.870 → 0.908 | 0.906 → 0.898 | 71% | 0.877 → 0.904 | True → True |
| glucose@48h | endocrine_stress | 49% | 0.877 → 0.880 | 0.908 → 0.902 | 25% | 0.851 → 0.848 | False → False |
| heart_rate@48h | cardiovascular_instability | 46% | 0.898 → 0.910 | 0.909 → 0.910 | 44% | 0.922 → 0.924 | True → True |
| map@48h | cardiovascular_instability | 46% | 0.890 → 0.907 | 0.906 → 0.904 | 44% | 0.841 → 0.846 | False → False |
| potassium@48h | electrolyte_acid_base | 68% | 0.896 → 0.899 | 0.908 → 0.887 | 62% | 0.854 → 0.847 | False → False |

External database already used: non-independent. No promotion. `causal_claim_allowed: false`.
