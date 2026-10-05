# Source-defined whole-state support

Unlike the earlier3feature current/age/rate bins, this diagnostic uses all
source-observed current values, masks and ages. Source patient median vectors
define center/std and nearest-other-patient90th-percentile distance threshold.
Evaluation query distance to these prototypes defines near/far groups, without
future outcome access. This geometry is not biological phenotype equivalence;
patient medians need not be actual simultaneous body states, and masks/ages
also reflect measurement processes. Not a learned support estimator.

Clean-full vs clean-own significant gain/harm out of7:
source near0/0, far0/2; hospital near0/0, far0/3.
Hospital near163-175patients, far42-56; some patients appear in both row groups.
Additional harm is present among geometrically farther rows, but the near
group still has no stable incremental advantage. It does not support the
claim that excluding only out-of-distribution rows restores whole-body gain.
Nor does it establish absence of information in the near group.

No new selector, training, gate relaxation, runtime, interval or promotion.
Artifact: `artifacts/clean_state_support_20260911.json`.
