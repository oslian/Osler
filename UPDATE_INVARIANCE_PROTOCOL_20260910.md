# Update Invariance and Feedback Attribution

Diagnostic only; glucose@6h, seven existing source seeds. Same source fit/tune
patients, all cached update anchors, frozen JEPA, and own-feedback training as
the final repaired experiment. No external tuning, runtime or interval change.

1. Documentation density: consume each encoded event 1, 2 or 4 times at the
same timestamp. Score only original anchors before extra copies; never create
extra labels, pending forecasts or error-memory updates. This is an exact
duplicate-ingestion stress test, not realistic extra fresh measurements. Compare
subsequent prediction shifts and patient-paired MAE differences at fixed weights.
2. Feedback attribution: fork the identical pre-anchor hidden/error state,
withhold only newly resolved errors for the current prediction, and discard
the fork afterwards. The normal path determines all subsequent history.
Stratify by whether a treatment snapshot changed between forecast origin and
feedback ingestion. This is local computational attribution, not a deployable
alternative policy or causal effect of treatment. Snapshot change includes
knownness and completed-dose transitions, not verified intervention changes.

Score the existing personalized subset (a prior outcome resolved). Report
patient-cluster paired bootstrap 95% CI, support and mean/p95 absolute shift.
1 mg/dL is a descriptive numerical sensitivity threshold, not clinical benefit.
Subgroup statistics are exploratory, not multiplicity-adjusted promotion tests.
Test replay parity with production, patient isolation, and delayed feedback.
The full raw-event ledger and storetime availability remain unverified.
