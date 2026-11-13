# Frozen-prediction loss alignment

Same native Mg checkpoints and roles. Source-tune patient-equal constant
offset fitted separately using weighted L1 median and Smooth-L1 optimum with
the model's normalization scale as native-unit beta. Both own and full corrected;
no evaluation-driven choice. This is NOT end-to-end L1 retraining or interval
calibration. It tests only a global offset explanation of objective mismatch.

| Median MAE | Own original | Own L1 | Full original | Full L1 | Full Huber |
|---|---:|---:|---:|---:|---:|
| Source evaluation | .217013 | .218257 | .218283 | .223025 | .219027 |
| Hospital | .182700 | .181913 | .184960 | .183869 | .184153 |

Full-L1 versus original gain0/7source,1/7hospital. Own-L1 versus original
gain0/7source,2/7hospital. Full-L1 versus own-L1 gain0/7both. Full-L1 versus
Huber gain0/7source1/7hospital, with source harm1/7. No stable incremental
repair from global MAE-aligned calibration; conditional loss effects remain
untested by this comparison. No runtime/promotion or new external lockbox.

Artifact: `artifacts/magnesium_loss_calibration_20260911.json`.
