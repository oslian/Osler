# Expanded insulin-potassium information diagnostic

Freeze seven existing patient/hospital partitions before fitting. Source tune
is unused: no hyperparameter search. Fixed histogram absolute-loss trees,
200 iterations, 15 leaves, minimum leaf 30, L2 10, learning rate .05.
Compare own potassium current/age and 6/24h known-event history, same history
plus insulin snapshot/sequence, and same history plus donor-patient context.
Fit patient-equal weights. No future elapsed-time input, no future treatments.
Use current plus signed residual; no amplitude clipping.

Report all rows and pre-query supported-sequence rows separately, source
evaluation and fixed hospital holdout, patient-equal MAE and patient paired
bootstrap against own, shuffled, persistence. Seven seeds share evaluation
patients; exploratory CIs are not multiplicity-adjusted independent trials.
Shuffled donor support may differ; a positive result requires follow-up controls.
No threshold adjustment from evaluation. No interval or deployment promotion.
This tests incremental information, not the full shared JEPA architecture.
Expanded cohort changes both support and composition; it does not isolate N.
Infusion timestamps are retrospective event-time, not proven knowledge-time.
No MIMIC or new pristine lockbox. Raw rows and fitted models stay private.
