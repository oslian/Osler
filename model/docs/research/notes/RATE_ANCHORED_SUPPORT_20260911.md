# Record-triggered query reconstruction

Rebuilt queries at unequal consecutive same-label insulin rate records, gap<=6h.
Conflicting equal-time entries removed; multiple simultaneous label-change
queries excluded. Only original paired-cohort stays/patients, with original
patient role assignment and hospital purge retained. No future values select
queries. Recompute current as-of lab and future5-7h measured pair at new query.

55468record queries/3198patients;5526paired/1763patients.4516windows/1526patients
have previous unequal record at/after current draw. Hospital521paired/193patients,
380bracket-after windows/146patients. Seed7source evaluation821bracket-after
windows/277patients. No training or accuracy claim.

Compared with lab-triggered strict group9hospital patients, the revised query
schedule restores support for testing record-triggered updates. It changes the
task/label distribution; do not compare unmatched MAEs as causal repair benefit.
Need identical new-query own versus contextual predictions. Availability of
infusion records still unverified, so retrospective timing only, not validated
online update or actual treatment change onset. No integrated dose inference.

Raw queries/labels/partitions private; aggregate report
`artifacts/rate_anchored_potassium_20260911.json`. No runtime/calibration/promotion.
An initial pandas read-only mask assignment error was corrected before completion.
