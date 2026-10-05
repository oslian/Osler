# Trained Observation-Process Control

Match expanded real training initialization, 39 source-eligible targets, samples,
eight epochs, optimizer, complete-exposure schedule and source-tune selection.
Only change: replace normalized numerical input channels with zero in BOTH
fit and tune from the beginning; preserve masks, ages, time and all remaining
channels. Real target labels unchanged. This is an equal-capacity process
control, not a treatment model or a new independent physiology model.

Use the same transformation at heldout evaluation. Compare selected process
with the already frozen selected real model and population median, with paired
patient errors. Preserve unsupported cells and nominal/multiplicity limitations.
No tuning using heldout results; no runtime promotion or input substitution.
Save separate optimizer/RNG checkpoints; historical real evidence unchanged.
