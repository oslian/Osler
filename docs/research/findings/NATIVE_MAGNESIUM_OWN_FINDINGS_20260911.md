# Magnesium own-history benchmark

Seven32epoch fits complete; identical evaluated pairs as real auxiliary model.
Both parent auxiliary and benchmark file bindings verified after execution.

| MAE | Own history | Multitask whole | Persistence |
|---|---:|---:|---:|
| Source evaluation | .2170 | .2234 | .2777 |
| Hospital | .1827 | .1814 | .2384 |

Whole-versus-own significant gain0/7both; source harm2/7, hospital harm0/7.
Both beat persistence7/7both. Whole's small hospital median advantage is not
evidence of robust extra-context gain. Joint auxiliary learning repairs weak
old outputs but has not beaten a properly supervised own-target benchmark.

Loss/selection differ explicitly: own uses magnesium-primary and magnesium
tune selection, whole glucose-primary with auxiliaryweight.25. This is a strong
task-specific benchmark, not equal-objective input ablation or proof other
variables contain no information. Source cohort/selection and loss imbalance
remain distinct hypotheses. No new runtime, calibration, lockbox or promotion.
Prior benchmark summary is refined, not retracted: both models truly beat
persistence, but the source of incremental whole-body value is unestablished.

Artifact: `artifacts/native_magnesium_own_20260911.json`.
Runner: `python -m scripts.train_native_magnesium_own`.
