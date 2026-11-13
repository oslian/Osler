# Auxiliary Future Supervision Pilot

Fixed three seeds7/19/31,2250 fitting and136 original source-tune patients.
Same availability-safe glucose queries, lab-only inputs, inherited weights,
source support masking, observed-step belief mask, identity decoder. Eight
epochs x16steps x32 patient-uniform draws, AdamW1e-4, weight decay1e-4,
gradient clip1. No MIMIC/hospital predictions in this source-only pilot.

Compare glucose-only versus glucose plus fourteen source-support-eligible
future targets versus shuffled-patient auxiliary deltas. Keep architecture and
primary glucose value+JEPA loss fixed. Add0.25 times the mean auxiliary
normalized value+JEPA loss over targets observed in the batch. Empty targets
produce no supervised term. Use existing target-specific heads/EMA encoders,
no new capacity. Use matched nominal6h query for5-7h observed labels; retain
actual label times in data, not as leaked future inputs.

Placebo preserves each target's fit delta distribution and availability mask;
sort fit rows by patient then circular-shift by the largest patient block to
forbid same-patient donors. Never borrow tune labels. This breaks correspondence
to the recipient's physiology; it is a supervision diagnostic, not a realistic
clinical alternative. It does not preserve all conditional label distributions.

Select checkpoint on source-tune glucose MAE including epochzero, identically
for each arm. Compare selected predictions with patient-paired bootstrap;
these reused selection patients are diagnostic, not an unbiased test. Positive
pilot evidence requires benefit over both controls, then a full7seed and heldout
trial, compatible interval calibration and retention before any promotion.
This pilot alone cannot authorize a runtime change, regardless of outcome.

This specifically tests supervision, NOT a new persistent organ trajectory.
Source support selected fourteen auxiliary targets without external outcomes.
Short budget cannot establish convergence; weak results cannot establish an
information ceiling. Candidate weights and patient predictions remain private.
