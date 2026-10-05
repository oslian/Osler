# E40: protect the actual shared-bridge update

## Question and novelty

E39 exactly reproduces E38 and locates most remaining local adverse directions
after momentum. This does not prove that removing momentum improves generalization.
Old bounded7 protected BUN descent/one-step replay and moment-clock experiments
already failed as general repairs. Do not repeat them.

E40 is a sustained intervention on the SAME E38 bridge and paired fit stream:
PCGrad before AdamW remains unchanged; only the actual resulting parameter step
is projected onto the intersection g1.dot(step)<=0, g2.dot(step)<=0 for pairs
of DIFFERENT targets. Same-target pairs are unchanged. This is a conditional
first-order constraint, not physiology, treatment causality or unseen safety.

Projection follows the convex half-space pattern used by
[GEM](https://arxiv.org/abs/1706.08840), not a claim to implement full GEM.
With only two constraints enumerate interior, each single boundary and the
intersection. PyTorch float64 least-squares supplies the intersection projection;
verify against SciPy constrained optimization in tests. Zero gradients impose
no constraint. No new library or custom Adam. Adam moments/step counter are
retained after correction, NOT reset or back-calculated. No line search.

Use nearest feasible step in float64, then round the proposed weight to the
original dtype. If that stored update violates either E38 local tolerance
1e-12*(1+||g||*||step||), retain the old weight and record a rounding veto.
This is numerical constraint enforcement, not a hidden rejection using future
or selection labels. Forward-evaluate the two training examples AFTER the step
for diagnosis only: these losses never choose a step/epoch/parameter.

## Fixed scope

All five E38 folds, same 2048 fit examples, two full passes, same seed20261004,
pair permutations, initialization, one 28672-parameter bridge, frozen backbone,
SmoothL1 beta1, clip1, AdamW lr1e-4/wd.01/defaultbetas. Fit labels only.
All53 target inventory, same fit/selection/same_source roles and20480 queries.
No target routing, new heads, epoch choice, learning-rate or tolerance sweep.
Original float64 event times, block masking and lawful inputs unchanged.

Primary comparator: saved, independently verified E38 PCGrad. Reuse E38 mean,
fixed_immediate, immediate, fixed_delayed, fixed_body, bound_start, unbridged,
frozen, raw. No weaker replacement; no re-fitting controls. Tests prove that
disabling correction precisely reproduces the original paired PCGrad trainer.
Bind previous artifacts, role rows, code, protocol, tests and saved new models.
Every non-bridge tensor must remain bitwise unchanged. Completed folds may be
resumed only when bindings match; never restart a live process after timeout.

## Acceptance and possible failure

Stage-level success: corrected different-target ACTUAL steps have zero locally
harmful dots under the fixed tolerance. Report raw-Adam versus applied harm,
step norms, projection/veto frequencies, and actual finite SmoothL1/MAE changes.
Do not conflate first-order guarantees and measured finite loss improvement.
Possible adverse effect: even successful local protection can slow fitting,
impede regularization or worsen other patients. Preserve same-target harm too.

Capability: candidate versus all TEN controls. Same E38 patient-collapse and
component weighting, paired bootstrap5000; seed20261004+targetindex*1000+
controlindex, component+100. Minimum30patients/3components; upper CI95<0 and
p_less<.05/(53*10) at both levels. Full screen requires all ten. Fit and
same_source descriptive; no posthoc restriction. Keep all unsupported entries.
Independent saved-model replay, scalar aggregation and screen reconstruction
required before recording completion. No release promotion from this screen.

If no full improvement, close this fixed actual-step branch. Local protection
alone does not resolve conditional transfer. This is current reconstruction,
not future forecast or self-learning deployment; MIMIC/NWICU are reused
development, no untouched lockbox. No runtime, intervals or causal changes.
Preserve10GiB disk; patient rows/models remain private, aggregate reports only.
