# Matched Broad-source Reconstruction

Three eight-epoch arms completed with536fit/90encoder-selection patients,
33original targets and141504updates/arm. Real/process/placebo select5/2/1.
Identical83874 current-value masked pairs from3433confirmation patients scored.
Six untrained targets remain explicit. This is not future forecasting.

Evaluator first failed before output: cached load_release model was expanded
twice, changing width178 to184. Independent deep copies repair this evaluator
bug; completed run checks disjoint parameter storage and reproduces old MAEs
to1e-10. Training checkpoints were not modified.

28/33targets improve pooled MAE over old. Target-equal normalized MAE:
old1.36561947, real1.30857997, population1.36309957,
process1.35723059, placebo1.36530386. About4.18% macro reduction.
Zero targets pass all fixed corrected patient/site comparisons against all four
references. ALT/AST/calcium/platelets beat the three non-old controls, but not
old. This partial signal is not a full development pass or promotion.

| Target | Old | Broad real | Broad population |
|---|---:|---:|---:|
| BUN | .970152 | .916954 | .988630 |
| Creatinine | 1.524346 | 1.454283 | 1.519404 |
| Bicarbonate | .956030 | .905453 | .938924 |
| Glucose | 1.265263 | 1.017556 | .999870 |
| WBC | 1.089224 | 1.017908 | 1.045425 |

These are normalized-unit errors, not clinical units. Glucose/lactate improve
old but still lose population/process/placebo. Bilirubin/fibrinogen/INR/platelets/
temperature worsen pooled old MAE. Source-selection support remains inadequate
for CPK/fibrinogen/lipase/PTT/triglycerides/troponin; do not erase these statuses.

Broader source improves candidate averages, not universal repair. Same people
counts/site quotas/update budgets do not control complete case mix, histories,
target support, source medians or chosen epoch. Selection is not proved the sole
cause. Separate population-prior correction from individual numeric information.
Only5-7sites contribute per target, so patient and site evidence differ.
Repeated fits, old-cohort retention and complete external gates remain pending.
Confirmation has been inspected; do not tune on it or call it a new lockbox.
Runtime/intervals/promotion unchanged. Patient rows stay Gitignored.

Reports: artifacts/native_reconstruction_matched_broad_training_20260917.json
and artifacts/native_reconstruction_matched_broad_evaluation_20260917.json.
