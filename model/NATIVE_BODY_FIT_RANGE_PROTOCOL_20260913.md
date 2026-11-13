# Training-Range Extrapolation Diagnostic

Repeat multi-scale reconstruction on identical patients. Before examining new
results, fix a uniform numerical-summary clipping rule: min/max of each feature
on the fitting patients only, all numeric features, no handpicked CPK threshold.
Support features unchanged. This is a transformed candidate representation,
not a replacement of observed clinical values. Training values are unchanged;
source-tune selects alpha/population using transformed tune inputs.

Compare unbounded, bounded, bounded patient-deranged placebo, support and
all-history incumbent with the existing paired bootstrap. Keep all patients.
Report max/error quantiles separately from average accuracy. Count CPK6h
two-event supports and actual span ranges across roles to localize the tail.
No tuning on heldout patients. Reused validation, no new lockbox/promotion.
Cost: range clipping can suppress genuine novel physiological extremes and
does not guarantee calibrated uncertainty or generalization.
