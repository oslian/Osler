# Expanded Current-Only Masked MAP Control

Removed future-pair requirement for current reconstruction. One fixed random
fresh<=1h MAP anchor per available seed7 patient, then independent-history mask.
Raw summary ridge versus support-only and patient-deranged placebo retain same
source-only alpha/scaler selection. This is not neural body training.

Heldout311patients: rawMAE12.27738 vs population12.00322, delta+.27417
CI[-.46234,1.12021]. Hospital194patients: raw14.24261 vs population14.30928,
delta-.06667 CI[-1.16092,1.07255]. No significant raw/population, raw/support
or raw/placebo advantage in either role. Placebo selects population.

Expansion removes the tiny53-fit-only and future-label-selection restrictions,
but this bounded raw summary model still shows no stable incremental signal.
It does not prove no signal exists, all sample sizes saturated, or encoder
irrelevance. Cohort changed; old/new MAEs not paired repair evidence. Input
history summary and one-anchor policy remain limitations. No promotion, runtime
or calibration changes. Exact postmask cohort counts are in report.

Evidence: `artifacts/native_body_masked_expanded_raw_20260913.json`.
