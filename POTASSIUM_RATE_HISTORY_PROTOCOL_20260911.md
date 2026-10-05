# Fixed rate-history diagnostic

Same source cohort, partitions, label and tree settings as infusion snapshot
trial.28 fits own/full x real/shuffled x7seeds. Append range, signed last-first
rate difference and observed span for each action/drug-compatible unit.
Only within(anchor-6h,anchor], at least2distinct times, one exact original
label, and no conflicting equal-time rates. No solution-volume inference,
cross-label merge, integration or assumed active duration. Zero retained.

Compare against frozen lab-only baseline and shuffled-context model with same
capacity; also report snapshot-trial MAEs as prior context (not independent
trials). Seven stable source gains required; no promotion/external reads.
Unknown temporal features stay missing. Uses retrospective event-time, not
independently confirmed availability. No hyperparameter tuning.
