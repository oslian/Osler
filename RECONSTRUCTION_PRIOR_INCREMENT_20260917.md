# Prior Versus Increment Diagnostic

All83874 saved pairs reproduce the matched broad-source scores. No fitting or
selection. Exact per-row identity:

new_error-old_error = new_center_error-old_center_error
  + (new_error-new_center_error)-(old_error-old_center_error).

Negative increments mean improvement over that model's own source-fit median.
This is an algebraic error decomposition, NOT an intervention isolating training
causes or proving the encoder has learned a biological mechanism.

| Target | Total change | Center-error change | Increment change | New increment |
|---|---:|---:|---:|---:|
| Creatinine | -.07006 | -.00474 | -.06532 | -.06512 |
| BUN | -.05320 | -.01364 | -.03956 | -.07168 |
| Bicarbonate | -.05058 | -.02518 | -.02540 | -.03347 |
| Glucose | -.24771 | -.07114 | -.17657 | +.01769 |
| O2sat | -.11413 | -.13861 | +.02448 | -.03856 |
| WBC | -.07132 | -.05929 | -.01203 | -.02752 |

Normalized units. Creatinine/BUN gains are not accounted for only by better
population centers. Both beat the new center descriptively in6/7sites, but the
full corrected development gate still fails. O2sat's center improves more than
the model; its incremental advantage shrinks. Glucose reduces both prior error
and excess model error yet remains worse than its center overall: improvement
is not a newly validated reconstruction capability. Lipase/triglycerides/troponin
and lactate similarly still lose their new centers. These are distinct failure
patterns, not a single universal absence of physiology or one hyperparameter.

No post-hoc gates, target removal or promotion. Three synthetic decomposition
tests pass, input hashes and old/new labels/sites agree. All39catalog statuses
retained; no runtime/interval changes. Next evidence must separate reproducible
numeric increment from fit variability and source support, using development
roles only for choices and preserving old-cohort retention checks.

Evidence: artifacts/native_reconstruction_broad_prior_increment_20260917.json.
