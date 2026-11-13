# Activity-Conditional Intervals (v4): MIMIC held-out check and NWICU re-check

- Cells: `38`
- Rule 1 (MIMIC held-out active and inactive coverage in [0.89, 0.91]): `21` of `37` cells → NOT met (needs ≥ 90%)
- NWICU full pass v2 → v4: `28` → `28` (non-independent) → rule 2 met
- v2 reproduces the v2 audit in every cell: `True`

| Cell | Module | MIMIC active share | MIMIC hold-out active v2 → v4 | inactive v2 → v4 | Ext. active share | Ext. patient cov. v2 → v4 | Full v2 → v4 |
|---|---|---:|---|---|---:|---|---|
| glucose@1h | endocrine_stress | 48% | 0.886 → 0.899 | 0.915 → 0.888 | 22% | 0.888 → 0.873 | False → False |
| platelets@1h | coagulopathy_heme | nan% | - → - | - → - | 83% | 0.910 → 0.904 | False → False |
| heart_rate@3h | cardiovascular_instability | 44% | 0.913 → 0.915 | 0.911 → 0.893 | 19% | 0.909 → 0.895 | True → True |
| hematocrit@3h | coagulopathy_heme | 79% | 0.890 → 0.900 | 0.920 → 0.906 | 80% | 0.915 → 0.919 | True → True |
| hemoglobin@3h | coagulopathy_heme | 87% | 0.905 → 0.909 | 0.914 → 0.892 | 79% | 0.910 → 0.907 | True → True |
| map@3h | cardiovascular_instability | 43% | 0.898 → 0.913 | 0.907 → 0.900 | 18% | 0.923 → 0.922 | True → True |
| platelets@3h | coagulopathy_heme | 89% | 0.887 → 0.886 | 0.925 → 0.897 | 84% | 0.905 → 0.896 | True → True |
| potassium@3h | electrolyte_acid_base | 68% | 0.896 → 0.908 | 0.912 → 0.898 | 63% | 0.904 → 0.907 | True → True |
| respiratory_rate@3h | respiratory | 52% | 0.908 → 0.906 | 0.911 → 0.898 | 34% | 0.910 → 0.902 | True → True |
| creatinine@6h | aki | 55% | 0.839 → 0.901 | 0.935 → 0.905 | 45% | 0.876 → 0.842 | True → False |
| heart_rate@6h | cardiovascular_instability | 42% | 0.880 → 0.884 | 0.902 → 0.903 | 18% | 0.887 → 0.892 | True → True |
| hemoglobin@6h | coagulopathy_heme | 86% | 0.908 → 0.903 | 0.932 → 0.893 | 78% | 0.917 → 0.905 | True → True |
| map@6h | cardiovascular_instability | 41% | 0.892 → 0.894 | 0.897 → 0.901 | 19% | 0.911 → 0.914 | True → True |
| potassium@6h | electrolyte_acid_base | 68% | 0.887 → 0.894 | 0.902 → 0.891 | 62% | 0.899 → 0.901 | False → False |
| respiratory_rate@6h | respiratory | 52% | 0.899 → 0.896 | 0.919 → 0.903 | 33% | 0.898 → 0.898 | True → True |
| bicarbonate@12h | electrolyte_acid_base | 75% | 0.889 → 0.899 | 0.938 → 0.914 | 65% | 0.876 → 0.868 | True → False |
| chloride@12h | electrolyte_acid_base | 74% | 0.878 → 0.899 | 0.908 → 0.903 | 65% | 0.896 → 0.904 | False → True |
| creatinine@12h | aki | 52% | 0.842 → 0.906 | 0.915 → 0.890 | 48% | 0.867 → 0.878 | False → True |
| heart_rate@12h | cardiovascular_instability | 45% | 0.880 → 0.898 | 0.894 → 0.894 | 19% | 0.882 → 0.888 | True → True |
| hematocrit@12h | coagulopathy_heme | 81% | 0.860 → 0.881 | 0.894 → 0.895 | 79% | 0.879 → 0.896 | False → True |
| hemoglobin@12h | coagulopathy_heme | 87% | 0.890 → 0.894 | 0.913 → 0.907 | 79% | 0.908 → 0.914 | False → False |
| map@12h | cardiovascular_instability | 45% | 0.914 → 0.913 | 0.905 → 0.897 | 19% | 0.903 → 0.897 | True → True |
| o2sat@12h | respiratory | 56% | 0.903 → 0.898 | 0.913 → 0.924 | 34% | 0.898 → 0.899 | True → True |
| potassium@12h | electrolyte_acid_base | 67% | 0.902 → 0.901 | 0.924 → 0.904 | 62% | 0.910 → 0.902 | True → True |
| respiratory_rate@12h | respiratory | 59% | 0.886 → 0.884 | 0.880 → 0.884 | 34% | 0.897 → 0.897 | True → True |
| bicarbonate@24h | electrolyte_acid_base | 71% | 0.897 → 0.890 | 0.943 → 0.913 | 59% | 0.899 → 0.883 | True → True |
| bun@24h | aki | 49% | 0.877 → 0.902 | 0.935 → 0.905 | 42% | 0.877 → 0.858 | False → False |
| creatinine@24h | aki | 49% | 0.884 → 0.902 | 0.926 → 0.894 | 41% | 0.913 → 0.903 | True → True |
| glucose@24h | endocrine_stress | 48% | 0.909 → 0.906 | 0.920 → 0.907 | 19% | 0.893 → 0.879 | True → False |
| heart_rate@24h | cardiovascular_instability | 44% | 0.898 → 0.898 | 0.887 → 0.892 | 20% | 0.894 → 0.891 | True → True |
| map@24h | cardiovascular_instability | 44% | 0.914 → 0.914 | 0.912 → 0.889 | 19% | 0.913 → 0.893 | True → True |
| potassium@24h | electrolyte_acid_base | 68% | 0.897 → 0.917 | 0.911 → 0.904 | 57% | 0.906 → 0.914 | True → True |
| respiratory_rate@24h | respiratory | 62% | 0.874 → 0.894 | 0.885 → 0.891 | 37% | 0.886 → 0.897 | False → False |
| sodium@24h | electrolyte_acid_base | 70% | 0.892 → 0.899 | 0.922 → 0.891 | 58% | 0.906 → 0.898 | True → True |
| creatinine@48h | aki | 53% | 0.870 → 0.908 | 0.906 → 0.898 | 45% | 0.901 → 0.916 | True → True |
| glucose@48h | endocrine_stress | 49% | 0.877 → 0.880 | 0.908 → 0.902 | 19% | 0.857 → 0.851 | False → False |
| heart_rate@48h | cardiovascular_instability | 46% | 0.898 → 0.910 | 0.909 → 0.910 | 20% | 0.905 → 0.901 | True → True |
| potassium@48h | electrolyte_acid_base | 68% | 0.896 → 0.899 | 0.908 → 0.887 | 58% | 0.902 → 0.893 | True → True |

External database already used: non-independent. No promotion. `causal_claim_allowed: false`.
