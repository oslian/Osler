# Bounded Site Sampling Protocol

Question: can source-only bounded site sampling improve shared reconstruction
without the extreme individual influence caused by exact site-uniform sampling?

For each source group/target, start from inverse site count. Multiply by a
common scalar and clip each patient's relative probability to [.5,2], solving
for mean weight one. Bounds fixed before training, not selected using outcomes.
Site identity only determines training probability, never model inputs. All
patients remain eligible. Caps bound expected influence, not realized random
repeat counts. Available but untrained targets remain explicitly pending.

Use identical with-replacement sampling for both a uniform-weight control and
the bounded candidate. Do not compare only against the old without-replacement
queue and attribute all differences to sites. Preserve initialization, seeds,
per-target/group draw budget, optimizer, masking and labels. Compare identical
checkpoints at fixed budgets and retain a common source-only selection rule.
Process/placebo controls remain required for promotion. No evaluation-derived
site weights, oracle target checkpoints, threshold changes or site exclusions.

## Fixed Matched Training

Both arms start at identical seed19 initialization and equal-group source-fit
median readout, not different pretrained checkpoints. Same 33 trained targets,
536 draws per target/group per epoch, eight epochs (283008 updates per arm),
AdamW1e-4/default betas/weight decay .01, SmoothL1, gradient clip1. Same event
inputs and masking. Existing selected90 and broad90 encoder patients choose
one common checkpoint per arm using the existing equal-group/target MAE;
epoch0 stays eligible. Broader source replay can be reported separately, not
silently change selection. Both arms sample with replacement using the same
seed/random-number consumption. Track realized unique patients and repeats.
Do not call this full source exposure or all39-target training.

This first comparison isolates source sampling for real-input candidates.
Matched new process/placebo training and full evaluation remain required before
any promotion. Previous process/placebo results cannot prove those new gates.
