# Expanded Shared-Body Training Protocol

Use all exact source fit/tune samples from the expanded native cache. Target
eligibility uses fit and tune >=30 patients only (39 targets), never heldout
performance. Same shared persistent body and fully target/sibling-masked inputs.

Two arms: real labels and target-wise cyclic patient-deranged fit labels.
Derangement preserves each target's label marginal and model capacity. Tune
labels remain real for both arms. Fix RNG7 scheduling and20260913 derangement.
Balanced round-robin consumes all fit pairs at least once each epoch, repeating
rare targets up to the largest target patient count. Both arms have identical
update budgets. Warm-start release, zero readout plus fit median, AdamW1e-4,
SmoothL1, clip1, deterministic eval-mode encoder, eight complete epochs.

Select each arm's checkpoint by target-equal normalized source-tune MAE,
including epoch zero. Persist optimizer, RNG and selected/current weights to
continue the SAME experiment at epoch boundaries. First execute through epoch1
as an end-to-end tranche; it is not sufficient training or promotion. Do not
inspect heldout accuracy until all eight epochs and source selection complete.

Compare selected real, selected placebo and fit median using patient-paired
errors in each evaluation role. Unsupported targets stay unsupported. This is
one source split and a reused MAP-derived cohort, not an untouched external
study or general proof of whole-body prediction. No runtime changes allowed.
