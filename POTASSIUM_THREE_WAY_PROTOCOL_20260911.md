# Whole-source three-way patient diagnostic

Freeze this protocol before training. Same potassium measured pairs and source
inputs as matched encoder. All eligible source patients, including expansion,
are hash-ranked by seed and divided60%fit/20%tune/20%evaluation. Every patient's
eligible queries belong to exactly one role. Seven existing seeds; masks saved
privately. Same roles for own/whole. No future values used to assign roles
beyond measured-pair eligibility; no clinical group or outcome stratification.

Same64epochs,16batches32, patient-uniform then query-uniform sampling, AdamW1e-4,
SmoothL1 plus EMA JEPA, identity value output, observed-step mask, warm start.
Source input support uses fitting patients only. Checkpoint chosen ONLY by tune
patient-equal MAE, including epoch0. Evaluation prediction performed only after
checkpoint selection, never at each epoch. Report all seven fits without using
evaluation to choose settings or seeds. Primary whole vs own patient-bootstrap
CI; persistence controls also reported. Evaluation is disjoint for this trial,
not a pristine lockbox: patients appeared in earlier experiments and warm-start
pretraining overlap is not excluded. No new external data or promotion.

Training count and composition differ from prior trials, so old-vs-new MAEs on
different populations cannot quantify a repair. This isolates incremental
context benefit within the new paired trial, not the causal effect of split
repair alone. No claim the finite64epoch budget establishes convergence.
