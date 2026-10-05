# E37: actual Adam gradient origins along the saved bridge fit

## Question and fixed scope

E36 establishes a locally harmful saved direction for 18 selection targets.
Did same-target gradients or gradients of other targets contribute that local
harm? Old 8--64-patient cosines and accumulation experiments do not answer
this question for E34's actual sequential Adam trajectory.

Replay the identical E34 immediate fit: all five folds, two passes, seed20261004,
AdamW lr1e-4/default betas/eps/weight_decay.01, SmoothL1beta1 and gradient clip1.
All original model tensors and training logs must reproduce exactly. No changed
hyperparameters, epoch choice or new efficacy model. Keep all53 target origins,
even if unobserved in a fold. Analyze all E36 20,480 queries across three roles.

## Momentum-aware decomposition

Carry a float64 first-moment vector m_j for each target j. At each original
optimizer step decay every m_j by beta1, then add (1-beta1)*clipped_gradient to
the current target's moment. Use the ACTUAL saved second moment v_t and actual
step to compute each origin's update -lr*m_j/(1-beta1^t)/(sqrt(v_t/(1-beta2^t))+eps).
Sum updates by origin across steps. Keep decoupled weight decay separately.
Keep actual float32 update minus reconstructed update as an explicit rounding
remainder; never hide it in one target. The resulting vectors must sum to the
exact final W (zero start) within1e-12. Report residual norms and moment errors.

Do not label the entire step by its current target: momentum from earlier
targets still contributes on that step. v_t, clipping, gradients and trajectory
are all jointly determined: origin decomposition is NOT a leave-task-out causal
effect, nor independently trained target weights. No origin deletion is deployed.

## Readout and weighting

At zero bridge compute grad_W p for each fixed query in float64. Project onto
all origin vectors, decay and rounding. Their sum must reproduce E36's saved
dp/dalpha within1e-9*(1+abs(s)). Starting output must reproduce E36 base64 within
1e-12. Original checkpoint unchanged. Directional MAE multiplier is
-sign(y-p0), except exact-zero residual uses sign(total slope); this assigns
contributions to the TOTAL ray's right derivative, not independent absolute
MAE derivatives for each origin. The zero-residual kink is not additive otherwise.

Collapse repeated folds within patient, then patient equal and source-component
equal; occurrence weighting secondary. Report full origin matrix, own/other/
decay/rounding contributions, unsupported targets and fixed near-zero tolerance
1e-10*(1+sum absolute contributions). Descriptive counts are not significance.
Compare fit, selection, same-source; all E36 supported inventory retained.
Independent scalar/fsum checks verify grouping and exact sums.

If own helpful but other harmful dominates, there is local cross-target
interference in this saved optimizer path, not proof of a universal repair.
If own harmful too, task isolation cannot be presumed sufficient. If both
patterns occur, retain that heterogeneity. No blanket gradient surgery,
posthoc target selection or new promotion from this diagnostic.

## Safety and verification

Tests cover momentum continuing on another target's step, zero gradients,
decay, nonfinite/invalid input, replay equality, all-origin sum, inference-mode
isolation, zero-residual nonadditivity and patient/component weighting.
Reuse existing PyTorch optimizer; do not rewrite its parameter updates.
Before results, verify weights bit-for-bit against E34, original logs, main
weights fixed and E36 query identities/truth. Bind all dependencies and outputs.
Save trace/fold/role outputs atomically, preserve10GiB free, no raw rows committed.
Numerical parity failures stop the run, never silently relax scientific gates.
No runtime, interval, forecast, causal or untouched-external validation claim.
