# Matched Broad-source Training

Freeze536fit/90encoder-selection patients and representative-site quotas before
training. Use the original33trained targets, seed19 release initialization,
source-fit median readout biases, observed_steps, real/process/placebo arms,
AdamW1e-4/default decay, SmoothL1, gradient clip1, eval-mode dropout and8epochs.

Exactly536draws per target per epoch:17688updates/epoch,141504/arm, matching the
original nested experiment. Shuffle each target's complete patient list then
cycle only after all examples have been used; target order randomized each round.
Source support differs and rarer targets are oversampled; report that limitation.

Select checkpoint by33target-equal encoder-selection MAE, with epoch0 eligible.
Record per-target scores; low-support targets may contribute to aggregate score
but never obtain individual validation eligibility through that inclusion.
Never load expanded confirmation/evaluation labels during training. Save every
epoch and restart only from hash-verified terminal checkpoint if needed.

Comparison controls total people, representative-site quotas and update budget,
not complete case mix, raw history counts, support or medians. It tests a broader
source candidate, not a unique causal explanation. All39catalog statuses retained.
No runtime, interval or promotion change without subsequent full gates.
