# Own-history event isolation check

Code-path audit: restrict_input_support clears non-target masks and values;
own training zeroes all six event features. observed_steps derives the step
mask from retained lab masks. NativeEventFilter adds only the observed-rate
flag, which is zero in own mode. NeuralBeliefFilter updates both belief and
previous_time only on active steps. Thus retained inactive timestamps do not
advance the own-history state clock.

Added a nonzero inactive-event regression test: arbitrary large contents and
different timestamps inserted before/between/after the same two observations
produce exactly identical state and uncertainty. Both native filter tests pass.
This is a structural check, not a new forecast comparison or proof of all-input
purity. Shared anchor selection remains potassium-selected and onset remains
available; no claim that cohort ascertainment effects are removed.

The suspected inactive-event cadence contamination is not supported by this
path/test. No training or runtime code changed. Do not remove valid target
remeasurements just because values match. Generalization remains unresolved.
