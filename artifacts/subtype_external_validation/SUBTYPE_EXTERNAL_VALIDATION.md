# Physiological Subtype External Validation

- Discovery: eICU only
- External lockbox: MIMIC-IV only
- Comparator: current release base point and patient-shuffled equal-capacity placebo
- Evaluated cells: 39
- Promotable cells: 0
- Promotable: none
- Runtime changed: no

| Cell | Base MAE | Subtype MAE | Placebo MAE | External pass |
|---|---:|---:|---:|:---:|
| bicarbonate@12h | 2.1899 | 2.3074 | 2.1881 | no |
| bicarbonate@24h | 2.51 | 2.4526 | 2.4576 | no |
| bun@24h | 6.7473 | 7.9344 | 6.8347 | no |
| chloride@12h | 2.7739 | 3.3462 | 3.1508 | no |
| creatinine@12h | 0.27177 | 0.4358 | 0.34342 | no |
| creatinine@6h | 0.1909 | 0.24814 | 0.21759 | no |
| glucose@12h | 33.823 | 34.283 | 35.331 | no |
| glucose@1h | 29.798 | 29.956 | 30.504 | no |
| glucose@3h | 29.741 | 30.052 | 30.847 | no |
| glucose@48h | 39.237 | 40.059 | 39.028 | no |
| glucose@6h | 31.261 | 33.37 | 32.972 | no |
| heart_rate@12h | 10.234 | 10.25 | 10.258 | no |
| heart_rate@24h | 11.166 | 11.247 | 11.245 | no |
| heart_rate@3h | 7.8657 | 8.0709 | 7.9395 | no |
| heart_rate@48h | 12.011 | 12 | 11.996 | no |
| heart_rate@6h | 8.917 | 8.9551 | 8.9665 | no |
| hematocrit@12h | 2.479 | 2.4389 | 2.4342 | no |
| hematocrit@3h | 2.1265 | 2.13 | 2.1391 | no |
| hematocrit@6h | 2.3121 | 2.3143 | 2.3126 | no |
| hemoglobin@12h | 0.81951 | 0.7999 | 0.79497 | no |
| hemoglobin@3h | 0.71908 | 0.73476 | 0.73374 | no |
| hemoglobin@6h | 0.79757 | 0.81853 | 0.80021 | no |
| map@12h | 10.658 | 10.536 | 10.665 | no |
| map@1h | 8.4446 | 8.4584 | 8.4401 | no |
| map@24h | 11.409 | 11.492 | 11.418 | no |
| map@3h | 9.7947 | 9.7611 | 9.8206 | no |
| map@6h | 9.6678 | 9.5737 | 9.8644 | no |
| o2sat@12h | 1.9667 | 8.1263 | 3.8891 | no |
| platelets@1h | 17.787 | 19.186 | 17.787 | no |
| platelets@3h | 18.247 | 20.719 | 18.555 | no |
| potassium@12h | 0.41909 | 0.49264 | 0.42755 | no |
| potassium@24h | 0.4063 | 0.40842 | 0.40935 | no |
| respiratory_rate@12h | 4.1245 | 4.1201 | 4.1128 | no |
| respiratory_rate@24h | 4.0989 | 4.1496 | 4.1885 | no |
| respiratory_rate@3h | 3.9553 | 4.0177 | 4.004 | no |
| respiratory_rate@6h | 3.8283 | 3.7654 | 3.8204 | no |
| sodium@24h | 2.8057 | 2.8208 | 2.8558 | no |
| urine_output@1h | 79.906 | 84.495 | 83.031 | no |
| urine_output@3h | 52.53 | 75.4 | 85.529 | no |

MIMIC labels were not used to fit the router, subtype priors, hyperparameters, or cell selection.
Only an exact cell that passes every listed gate is eligible for later runtime integration.
