# Initial gradient response to repair

Seven source partitions, first3sampler batches each, all evaluated at the SAME
initial parameters without optimizer steps. Same clean labels and selected
patient rows; old/clean input contracts use their source-fit transforms.
These are21initial probes, not a replay of3training updates or later trajectory.

Whole gradient median relative change is0.013329 and cosine0.999951.
History/mixer/sharedfuture/private block median relative changes are0.012882,
0.013291,0.012463,0.010582. Their median cosines all exceed0.9999.
Thus the repair does not produce a large typical initial gradient reversal in
these probes. Nonlinear accumulated effects and Adam state remain untested;
small initial gradient change can still yield different later checkpoints.
Do not label this optimizer instability or convergence failure as proven.

This narrows the prior proposed gross-gradient-defect explanation without
repairing the whole-context generalization gap. No optimizer steps, training,
runtime, calibrator or promotion changed.
Artifact: `artifacts/native_conflict_initial_gradients_20260911.json`.
