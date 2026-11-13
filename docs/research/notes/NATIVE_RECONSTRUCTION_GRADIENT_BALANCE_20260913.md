# Initial Multi-Target Gradient Diagnostic

Eight source-fit patients,1,877anchors,6,249 current reconstruction examples.
Reservoir sample one example per patient/target using fixed seed20260913 gives
232examples across38targets. No fitting/update/evaluation was performed.
Each target's mean patient gradient is measured on shared body-update-gate
parameters at the same initial model. Other parameter groups are not audited.

HR/MAP/O2sat/RR comprise64.394% of available examples. Under a row-count weighted
approximation to joint training, HR/O2sat/glucose have the largest weighted
gradient magnitudes. This approximation weights patient-mean sampled gradients
by target event counts; it is NOT the exact gradient of all6,249 rows.
Target-balanced gradient norm .021319 vs event-weighted .010714; larger norm
is not proof of a better update. Pairwise gradient cosine median .001738,
48.65% negative. Random newly initialized body/readout modules can produce
near-orthogonal/opposing gradients; this is not evidence of learned destructive
interference or the root cause of past JEPA failures.

Loss size alone does not identify domination: glucose mean loss6.283 versus
O2sat.200, yet its body gradient is only about2x O2sat; SmoothL1 limits large
residual derivative. Do not introduce loss balancing solely from loss ratios.

Concrete training implication: preserve per-patient/per-target sampling and
report target-wise losses, rather than blindly replay every measurement as
an equally informative example. Test actual heldout performance before any
gradient-conflict surgery or promotion. No guarantee for targets with1patient.

Evidence: artifacts/native_reconstruction_gradient_balance_20260913.json.
No weights saved, runtime/calibration/promotion changes. Full training pending.
