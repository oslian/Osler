# Fit-Only Kernel Scale Repair

The previous raw RBF metric was dominated by hour-valued measurement lag.
Test one intervention: weighted StandardScaler fitted ONLY on fitting inputs,
before the unchanged kernel map and absolute-error regressor. Use existing
scikit-learn StandardScaler, population variance, constant-feature scale one.
No clipping, whitening, covariance fitting, feature removal or parameter search.
Missingness remains encoded, not replaced with fictitious observations.

All47 targets/all5 original source folds. Four arms scaled_body/scaled_raw/
scaled_process/scaled_shuffle, same frozen encoder, true labels, target/sibling
masking, bounded component weights, pair draws,128landmarks, SVR settings and
within-component shuffled-body control as the prior kernel experiment.
Process control zeros the first106 raw numbers before fitting its scaler.
Body shuffle uses the SAME fitted scaler/map as real body, not a new map.
Fit scalers never inspect query rows or labels. Constant features are not
deleted; query-only variation is reported and no range clipping is performed.

Prespecify raw and body efficacy separately. Each candidate must beat native,
body HGB, raw HGB, population, its corresponding unscaled kernel, scaled_process
and scaled_shuffle. One family covers2*47*7 comparisons. Retain patient-equal
fold-first and component-equal errors,>=30patients>=3components, paired5000
bootstrap upper95%<0, one-sided t p<.05/family, negative mean at BOTH levels,
and all fitting solvers converged. Report every target, not a winning subset.
These are stricter joint diagnostic screens, not runtime promotion gates.
Report old seven-comparator results only from frozen old artifacts, unchanged.

For scale repair, decompose identical fit-pair distances before/after scaling,
including numeric removal kernel effect. Report constant-feature count and
maximum query magnitude in standardized coordinates. Large scales for rare
features/outliers may hurt; improvement is not presumed. Rotation check applies
only AFTER standardization, simultaneously rotating landmarks and inputs. This
is NOT original-coordinate rotational invariance: per-coordinate standardization
defines a new metric. Test unit-rescaling equivariance, fit/query isolation and
unchanged landmarks explicitly. Keep all original raw/body HGB comparators.

Do not read outer datasets, change serving/checkpoints/calibrators, infer causal
physiology, or tune on reused source selections. No new package. Use published
library pattern https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html
Already-inspected source results are development, not an untouched lockbox.
