# Expanded Confirmation: Fixed Nominees

Completed frozen real-encoder inference for83874patient-target pairs across33
trained targets. All39statuses reported:28have fixed nominees,5have original
diagnostic predictions but no supported nomination,6have no trained source head.
New labels were not used for fitting, alpha choice or threshold selection.
Selected encoder tensors match the previous feature cache exactly.

Under the unchanged patient/site tests against original and fit-median with
.05/39 Bonferroni threshold, zero nominees confirm. Nine are numerically better
than original, including two population-only nominees(creatinine/lactate), which
cannot establish extra physiology. No promotion or runtime change.

| Target | Patients | Original normalized MAE | Nominee MAE | Interpretation |
|---|---:|---:|---:|---|
| BUN | 3318 | 0.97015 | 1.04924 | Harm in patient and equal-site means |
| Bicarbonate | 2947 | 0.95603 | 0.98252 | Earlier small confirmation gain does not reproduce |
| WBC | 3355 | 1.08922 | 1.13370 | Harm persists with larger support |
| O2sat | 3382 | 0.94262 | 0.85451 | Mean benefit; site p vs original0.00550 exceeds0.001282 |
| Phosphate | 1968 | 0.69776 | 0.68196 | Pooled benefit but equal-site delta+0.02249 |
| Respiratory rate | 3391 | 0.93028 | 0.92245 | Small benefit, insufficient site evidence |

This distinguishes observed adverse predictions from insufficient evidence of
benefit. More confirmation patients clarified the result; they did not improve
the fixed models. Seven original confirmation hospitals remain, so number of
independent sites is not increased. Site weighting still includes small sites;
do not treat sign flips alone as proven physiological domain change.

The expanded population also differs from the original query-selected cohort.
Same hospitals do not guarantee matched case mix or observation histories.
No unique root-cause claim: source-fit representativeness, finite-head estimation
and body representation remain competing explanations. The next diagnosis should
compare original versus expanded confirmation input/label support without tuning
new heads on this inspected confirmation set. Simply relaxing the selector is
contradicted by BUN/bicarbonate/WBC actual harm.

Six confirmation/selection helper tests passed. Every scored target has exactly
the accepted sample count and one row per patient. Private per-patient predictions
and hashes retained; no patient IDs in aggregate artifacts.

Evidence: `artifacts/native_reconstruction_expanded_confirmation_evaluation_20260917.json`.
These are current-value reconstruction results, not forecasts, clinical units,
calibrated intervals or untouched external validation.
