# Source joint-support diagnostic

Fixed original native glucose predictions. Joint strata use current glucose,
measurement age, and last finite recorded insulin rate in the48h history.
Per-seed source fitting patients alone define quartile cutpoints, using one
median per patient; joint-cell support counts distinct fitting patients.
At least30 source patients is the prespecified descriptive support split, not
a learned routing threshold. No label/outcome defines these strata.
Aggregate: `artifacts/glucose_native_support_20260911.json`.

| Evaluation stratum | Patient range across splits | Whole-own significant gain/harm |
|---|---:|---|
| New source, under30 fit support | 113-128 | 0/7,2/7 |
| New source, at least30 | 268-310 | 0/7,1/7 |
| Hospital, under30 | 65-85 | 0/7,1/7 |
| Hospital, at least30 | 160-172 | 0/7,0/7 |

Patients may appear in BOTH strata at different queries; counts must not be
added or treated as independent cohorts. Patient bootstrap applies within
each stratum; overlapping seeds are not independent replications.

Even the supported stratum does not show reliable incremental whole-context
benefit. This does not support blaming failure solely on rare combinations
of these three measured features. It does NOT show adequate support for the
full high-dimensional patient state: coarse bins can hide heterogeneity and
tails. Last recorded rate is not verified ongoing delivery; freshness, dose,
route and actual administration semantics are not established by this feature.

No low-support abstention adopted, no tuning on hospital outcomes, no neural
fit, runtime, interval or promotion change. All original hashes verified.
This is a stratified diagnosis, not a completed remedy or new external lockbox.
