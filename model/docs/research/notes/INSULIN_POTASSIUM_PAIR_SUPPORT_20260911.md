# Expanded matched pair support

Full-table insulin-rate stay inventory matched to strict canonical potassium
labs using existing availability reconstruction (max draw/final revision time).
Queries at potassium available times; current age<=6h; nearest future draw
5-7h after query. Same-label nonconflicting insulin sequence in previous6h.
No outcome magnitude used for eligibility; this is support, not predictive fit.

| Stage | Rows | Patients | Hospitals |
|---|---:|---:|---:|
| Potassium availability queries in rate-recorded stays | 89156 | 5590 | 70 |
| Paired current/future potassium | 12354 | 3605 | 68 |
| Pre-query sequence | 21309 | 4201 | 64 |
| Both paired and sequence | 3373 | 1562 | 54 |

The previous small evaluation subgroup is not a hard raw-data ceiling.
Matched expansion is feasible. No evidence yet that it improves forecast.
Must establish source fit/tune/evaluation/hospital roles before training and
keep patient identity grouped across stays. Previously inspected hospitals
are not pristine lockboxes. Cohort defined by ever having an insulin record
is a descriptive sampling frame, not a prospective clinical eligibility rule.

Treatment event-time still lacks independent availability; do not label this
availability-safe treatment deployment. Labs use conservative final-revision
policy, not an assertion of first result visibility. Shared outcomes across
queries are not independent labels. Window counts do not replace patient counts.

Only aggregate report in repo. Canonical labs, query identities and labels stay
under private tmp; hashes bind them for subsequent prespecified development.
Artifact: `artifacts/insulin_potassium_pairs_20260911.json`.
No training, external outcome evaluation, calibration, promotion or runtime change.
