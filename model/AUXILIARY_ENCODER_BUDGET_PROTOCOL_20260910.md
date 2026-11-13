# Fixed Longer Auxiliary-Supervision Trial

The8epoch pilot did not show significant benefit, but all glucose-only and
real-auxiliary selections were at epoch8. This follow-up tests that budget
ambiguity, not a declared positive pilot. Freeze64epochs and the existing seven
seeds7/19/31/43/59/71/83 before reading longer training results.

Keep every pilot label, input, fit/tune role, architecture, initialization,
optimizer, sampling policy, loss and auxiliary coefficient unchanged.
Three arms per seed: glucose-only, real auxiliary and shuffled auxiliary.
No external data evaluation. Exactly2250fit/136tune patients; select checkpoint
on source-tune glucose MAE, including epochzero. No early stop/restart based on
intermediate scores. All21 fits must finish. Candidate artifacts stay private.

Require all first8logs to exactly reproduce the pilot for its three seeds and
all64 glucose-only tune scores to reproduce the existing2250/64 whole-unbounded
experiment. Bootstrap patient-paired error differences against both controls,
1000 resamples. Report all seven results, including harm and null outcomes.
These selection-data CIs are diagnostic, not external promotion evidence.

No threshold, target group, weight, or checkpoint may be selected using MIMIC.
Even source-side success requires independent hospital evaluation, own-history
comparison, negative-transfer checks and compatible calibration. A negative
result rejects this fixed supervised extension, not all possible multi-organ
learning;64epochs is not automatically convergence. This does not add a
persistent organ dynamics architecture or alter the runtime model.
