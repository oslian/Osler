# Native magnesium dropout results

Seven32epoch fits complete. Fixed50%batch own-only augmentation, full inputs
for selection/evaluation, independent augmentation RNG. Parent/output hashes
verified. This transfers a previously tested glucose idea, not a novel method.

| Median MAE | Matched whole | Dropout | Own |
|---|---:|---:|---:|
| Source evaluation | .218283 | .216822 | .217013 |
| Hospital | .184960 | .183796 | .182700 |

Dropout versus matched whole and own has0/7significant gain,0/7harm in both
roles. Persistence7/7both. Small numerical changes do not establish repair.
Selected epochs21,6,13,24,31,24,4. No general conclusion that all regularization
is ineffective; no external setting selection, calibration or runtime promotion.
Artifact: `artifacts/magnesium_context_dropout_20260911.json`.
