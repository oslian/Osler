# Final State Update Trace

Frozen bounded7,525 source-encoder examples, at most8people per target/group.
Same33trained targets, target+siblings masked as before. Final predictions from
explicit replay reproduce the normal model within1e-7 on every example.
Trace exposes before-last-advance, before-last-observation-update,
after-last-update and final-after-gap readouts of the SAME trajectory.

| Stage error change | Selected macro | Broad macro | Targets with higher error selected/broad |
|---|---:|---:|---:|
| Last advance | +.001205 | +.004846 | 18/18 |
| Last observation update | -.046327 | -.153547 | 15/11 |
| Final gap advance | +.000439 | +.000338 | 5/5 |

Negative means lower normalized anchor-truth error. All intermediate readouts
are retrospectively compared to the final anchor truth. These are not scored
forecasts at earlier times and do not prove accurate dynamics or causal effect.

Creatinine final observation update increases mean error+.072347 selected and
+.047543 broad (8patients each, zero final gap). BUN increases+.172566 selected
but decreases-.422583 broad; glucose reverses directions too. Lipase mixed,
with only7broad people. No significance or universal forgetting claim follows
from these small, checkpoint-selection samples.

## Decision

The last measurement update is not uniformly destructive; its average benefit
exceeds the last advance effect on these probes. The creatinine location is a
specific follow-up hypothesis, not justification for an oracle skip-update gate.
Before changing state-update architecture, expand source-only paired checks and
inspect what legal observation content triggers the update. Do not condition
deployment on whether the hidden target later turned out better or worse.

Trace tests cover zero/nonzero anchor gap and removal of target-only last events.
No fitting, deletion of updates, new weights, runtime or promotion changes.
Evidence: artifacts/native_reconstruction_final_update_20260920.json.
