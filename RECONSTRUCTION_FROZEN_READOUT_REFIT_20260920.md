# Same-Capacity Frozen Body Readout Refit

## Comparison

Current bounded7encoder, not the older20260917raw-state experiment. Extract
the exact masked-current body vector for all14,150sourcepatients:247,611fit
and77,219encoder-selection patient-target pairs. Verify every prediction and
truth against prior caches. Preserve target/sibling block masking, timing,
one patient-target observation and independent source roles.

For each of33targets, refit the SAME linear64-dimensional readout using weighted
SmoothL1(beta1), expected bounded-site weights and equal source-group mass.
Feature centering/scaling is an invertible optimization coordinate change only;
weights are converted back to native coordinates. Initialize at the original
head. L-BFGS-B max2000steps,ftol1e-12,gtol1e-7, no selection-tuned regularizer.
No extra capacity, no second body state, no encoder updates. Six untrained rows
remain unchanged. This is data-risk optimization, NOT AdamW-equivalent fitting.

All33solvers report convergence in99-198iterations. Maximum absolute scaled
gradient2.742e-6 (termination can use function tolerance). Do not claim exact
zero gradients or globally optimal end-to-end JEPA. Each fitting loss improves.
Target-equal fitting SmoothL1:1.258742272->1.190608141 (~5.41%reduction).
Tensor audit confirms only readout.weight/readout.bias changed.

## Source-Held-Site Results

| Group | Original target-equal MAE | Refit | Numerical gains /33 | Incremental passes |
|---|---:|---:|---:|---:|
| Selected90 |1.797984335|1.896873820|14|0|
| Broad3323 |1.559770689|1.601832406|9|0|

Full broad source-selection, not the old90person broad subset or separate
evaluation cohort. Do not compare these macro values across differently sampled
reports. Retain existing patient/site p<.05/39 incremental gate.

BUN improves in mean bothgroups: selected1.105465->1.054067, broad.893612->
.821318. Selectedpatientp.1048/sitep.0457; broadsitep.002144>.05/39, so neither
passes. Overall training improvement accompanies aggregate selection regression.

## Diagnosis And Decision

A readout optimization gap exists, but simply solving the existing linear
readout more thoroughly does not provide a generalizable repair. This rejects
the narrow 'only insufficiently optimized head' explanation for overall failure.
It does NOT prove no transferable information exists in the body state: feature
conditioning, regularization, site/case-mix shifts and finite support remain
possible. No blind new-head or capacity sweep is justified by this result.

The cached current body features now permit direct conditioning/rank and
source-held-site error diagnosis without another expensive neural replay.
Review previous ridge/regularization studies before proposing changes; their
older encoder/cohort cannot be silently equated to this current candidate.

Two analytic-gradient/affine-recovery tests pass; full extraction and solvers
exit0, exactcache/role checks pass. Candidate remains private. No evaluation
promotion, release weights, intervals, registry or runtime modified. This is
current masked reconstruction, not future forecasting or clinical validation.
Evidence: artifacts/native_reconstruction_frozen_readout_refit_20260920.json.
