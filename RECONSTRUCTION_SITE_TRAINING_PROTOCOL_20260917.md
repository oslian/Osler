# Fixed Site-Partition Reconstruction Comparison

Use the frozen common partition from native_reconstruction_site_partition_20260917.
All33fit/tune-supported targets train jointly; report the other six explicitly.
536fit and179tune patients;178evaluation and870reserved patients NEVER enter
optimization or checkpoint selection. This is development evidence on reused
data, not a fresh external lockbox or certified independent pretraining.

Same existing NativeBodyReconstruction/NativeOrganStateResidual warm start,
observed_steps input, six zero event channels, full target/sibling history masks.
Readout zero weights and fit-only median biases. Seed19 initialization/sampling.
Real numeric, trained process-only and deranged-patient placebo use the same
architecture, target-balanced complete-exposure schedule and eight-epoch budget.
Process neutralization applies in BOTH fit and tune. Placebo fixed donor mapping
uses seed20260913 and original actual tune truth. AdamW1e-4, default decay,
SmoothL1, gradient norm clip1, eval-mode dropout off, four CPU threads.

Each arm selects minimum target-equal tune MAE including epoch0. Save complete
optimizer/RNG state each epoch to durable private storage. Intermediate epoch
boundaries may be resumed without changing schedule or stopping policy. Do not
inspect evaluation until all three arms finish epoch8. No per-target selection
using evaluation, threshold relaxation, or comparison of changed-cohort MAEs
as if they were paired repair gains. Original population/process/placebo tests
remain separate from full release promotion, intervals and future forecasting.
