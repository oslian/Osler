# Native MAP Content Control

## Question

On the rebuilt MAP availability-native task, do other numerical observations
help beyond target history and their measurement support? Prior glucose
bedside support and inference-only MAP update-scope experiments do not answer
this matched-training question on the new MAP anchors.

## Fixed Comparison

Use the original seven native-query partitions, source-only transformations,
initial weights, patient sampler, 32 epochs, optimizer and JEPA/value loss.
Compare original full against a new support-control arm trained from the same
initialization. The control retains MAP numerical history, all observation
masks, ages, module presence, timestamps, and event observed flags. It zeros
non-MAP normalized numerical values and the three treatment numerical fields
in history, and non-MAP numerical query values. No future value selects inputs.

The number and timing of active updates must exactly match full. Padding must
remain inactive, real zero values retain their original masks, and raw/full
arrays must remain unchanged. This is not a pure cadence placebo: variable
identity, freshness and measurement process remain available.

Compare against full, own and persistence using the same patient-equal MAE
and paired bootstrap. Source-tune selects checkpoints. Evaluation/hospital
results are reused diagnostic sets, not a new lockbox. No automatic promotion
or modification of release weights or calibrators. If control is better than
full, that implicates numerical content conditional on this support contract,
not a proof that all other-organ physiology is useless or causally harmful.

## Current Status

Numeric-control transform implemented. Seven unit tests covering this helper,
native activation and compact batches pass. The new control has NOT yet been
trained or accuracy-evaluated. Training runner and full-data equivalence checks
remain required before efficacy claims.
