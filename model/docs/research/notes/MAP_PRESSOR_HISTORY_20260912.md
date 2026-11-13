# Recorded Pressor History Support

Built private fixed past-cutoff snapshots at anchor-1h and anchor-3h with the
same6h recorded lookback and conflict policy. Current snapshots reproduce
previous cache exactly. Signed differences require same drug/dimension and
strictly later recorded source time; repeated carry-forward is not a new pair.
Eight tests pass, including signed decrease, zero remeasurement, missingness,
prefix invariance and conflict handling.

Norepinephrine1h comparison supports6,455queries, with2,687changed queries
across494patients. At3h,6,760pairs,4,100changed queries/531patients.
Vasopressin1h has1,691pairs,143changed/78patients;3h1,735pairs,
246changed/95patients. Other drugs are in the aggregate report. Counts overlap.

This establishes that prior current-only rate snapshots omit recorded history
variation; it does not establish that this information predicts future MAP.
These are comparisons at past cutoffs, not complete administration/start/stop
histories. Each snapshot can be up to6h old; actual source-time gap is retained.
Independent availability remains unverified, hence online_eligible=false.

No training, accuracy or promotion claim. Current model/registry unchanged.
Evidence: `artifacts/map_pressor_history_20260912.json`.
