# Broad-source Model Retention

Fixed matched-broad checkpoints evaluated on original178evaluation patients,
4637 identical masked current-value pairs. Fit/encoder patient overlap is zero.
Parent prediction/checkpoint bindings and original sample hashes verified;
labels exactly agree with original cached truths. No fitting or head selection.

Only5/33targets improve pooled old-cohort MAE;28worsen numerically.
Target-equal normalized macro MAE:old2.93423214,new3.00741985 (+2.49%).
This contrasts with28/33improving in expandedconfirmation and macro-4.18%.
Not all numeric harms are significant; absence of significance is not proof of
noninferiority. Per-site results are noisy and often differ from pooled means.

| Target | Old | Broad model |
|---|---:|---:|
| Glucose | 1.65141 | 1.95444 |
| Bicarbonate | .97608 | 1.17855 |
| BUN | .91290 | .94767 |
| Creatinine | 1.45341 | 1.46187 |
| Albumin | .64283 | .78299 |

All normalized units. Replacing the selected cohort improves one distribution
but does not preserve the original distribution's accuracy. This is a measured
tradeoff, not a universal repair or proof cohort membership causally determines
error. Case mix, hospital composition, support and sample exposure still differ.
Do not route using retrospective qualifying insulin changes: that would require
future information. Need broad training that retains relevant state support and
source-only validation of both groups, not confirmation-driven switching.

No runtime/interval/promotion change; all39statuses retained. Same fixed training
is reused, not an independent fit or untouched external validation.
Evidence: artifacts/native_reconstruction_broad_retention_20260917.json.
