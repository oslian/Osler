# Stable, Change and Uncertainty Replay

Frozen release diagnostic, not clinical validation or promotion. No fitting or threshold selection.
eICU and NWICU have been used in prior development; these are NOT untouched lockboxes.
Sampling is one eligible anchor per patient per cell, selected by hash, not by change size.
Actual future change defines retrospective strata only. The model never receives these strata.
Stable means smaller than a research effect size, NOT healthy or safe. Thresholds are not alarms.
Intervals are exactly source-bound release intervals, never refitted for stable/change subgroups.
CIs use equal-patient bootstrap. A subgroup needs 30 patients for a CI. No multiple-testing promotion.
Domain slices are descriptive replays, not new hospital-heldout training experiments.
Missing intervals/points are explicitly counted; coverage is conditional on issued intervals.
The central interval score penalizes width and misses; it is not CRPS or full-distribution validation.
Binary hypoxemia is not replaced with single-time O2sat: its interval-event labels need a separate audit.

| Database | Cell | Patients stable / changed | Stable false move | Changed signed recall | Changed MAE delta | Coverage stable / changed |
|---|---|---:|---:|---:|---:|---:|
| eicu | glucose@1h | 266 / 234 | 0.056 | 0.141 | -3.899 | 1.000 / 0.778 |
| eicu | map@1h | 300 / 200 | 0.020 | 0.070 | -2.277 | 0.993 / 0.665 |
| eicu | platelets@1h | 244 / 256 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | urine_output@1h | 243 / 257 | 0.078 | 0.253 | -51.683 | 1.000 / 0.718 |
| eicu | glucose@3h | 244 / 256 | 0.102 | 0.148 | -5.788 | 1.000 / 0.824 |
| eicu | heart_rate@3h | 373 / 127 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | hematocrit@3h | 361 / 139 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | hemoglobin@3h | 350 / 150 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | map@3h | 292 / 208 | 0.024 | 0.091 | -2.757 | 1.000 / 0.769 |
| eicu | platelets@3h | 274 / 226 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | respiratory_rate@3h | 192 / 308 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | urine_output@3h | 251 / 249 | 0.054 | 0.245 | -33.041 | 0.985 / 0.663 |
| eicu | creatinine@6h | 412 / 88 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | glucose@6h | 244 / 256 | 0.139 | 0.230 | -13.604 | 0.996 / 0.801 |
| eicu | heart_rate@6h | 350 / 150 | 0.009 | 0.020 | -1.808 | 1.000 / 0.682 |
| eicu | hematocrit@6h | 330 / 170 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | hemoglobin@6h | 338 / 162 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | map@6h | 249 / 251 | 0.068 | 0.124 | -3.330 | 1.000 / 0.697 |
| eicu | respiratory_rate@6h | 191 / 309 | 0.079 | 0.207 | -1.361 | 0.986 / 0.644 |
| eicu | bicarbonate@12h | 199 / 301 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | chloride@12h | 293 / 207 | 0.000 | 0.000 | -0.489 | NA / NA |
| eicu | creatinine@12h | 411 / 89 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | glucose@12h | 209 / 291 | 0.335 | 0.368 | -14.512 | 0.995 / 0.866 |
| eicu | heart_rate@12h | 298 / 202 | 0.027 | 0.064 | -2.465 | 1.000 / 0.817 |
| eicu | hematocrit@12h | 354 / 146 | 0.016 | 0.027 | -1.335 | 0.968 / 0.622 |
| eicu | hemoglobin@12h | 344 / 156 | 0.038 | 0.019 | -0.378 | 0.981 / 0.652 |
| eicu | map@12h | 237 / 263 | 0.101 | 0.194 | -4.291 | 1.000 / 0.779 |
| eicu | o2sat@12h | 357 / 143 | 0.000 | 0.000 | -1.062 | 1.000 / 0.699 |
| eicu | potassium@12h | 255 / 245 | 0.127 | 0.171 | -0.182 | 0.983 / 0.778 |
| eicu | respiratory_rate@12h | 160 / 340 | 0.237 | 0.344 | -1.369 | 1.000 / 0.715 |
| eicu | bicarbonate@24h | 218 / 282 | 0.000 | 0.014 | -0.167 | 1.000 / 0.923 |
| eicu | bun@24h | 284 / 216 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | heart_rate@24h | 288 / 212 | 0.035 | 0.104 | -2.810 | 0.997 / 0.792 |
| eicu | map@24h | 214 / 286 | 0.164 | 0.231 | -3.707 | 0.991 / 0.713 |
| eicu | potassium@24h | 241 / 259 | 0.165 | 0.212 | -0.158 | 1.000 / 0.884 |
| eicu | respiratory_rate@24h | 165 / 335 | 0.406 | 0.466 | -1.891 | 1.000 / 0.746 |
| eicu | sodium@24h | 289 / 211 | 0.000 | 0.005 | -0.351 | 1.000 / 0.817 |
| eicu | glucose@48h | 210 / 290 | 0.000 | 0.000 | 0.000 | NA / NA |
| eicu | heart_rate@48h | 219 / 281 | 0.087 | 0.224 | -4.585 | 0.995 / 0.858 |

MAE delta is model minus persistence (negative is better), in physical units.
Recall requires a threshold-sized prediction in the correct direction; abstentions count as misses.
Unsupported cells and all half/double-threshold sensitivities are recorded in report.json.
The replay is conditional on a measured future label; it does not establish performance when no test occurs.
No treatment-intent or future-label information is added to inference. No next-measurement-time model.
