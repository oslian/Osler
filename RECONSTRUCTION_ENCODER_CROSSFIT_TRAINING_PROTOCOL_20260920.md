# Encoder-Independent Reconstruction Protocol

Scope: all 33 trained current-reconstruction targets, all five source-hospital
component folds defined by the completed preflight. This is not future JEPA
forecast training or an external lockbox. No serving changes are authorized.

## Fixed Comparison

Train two same-capacity arms from identically seeded fresh parameters:
real observations and process-only observations. Process-only zeros physiological
numbers after fold-local scaling, retaining the same event clock, masks and
ages. Both use the existing NativeBodyReconstruction / NativeOrganStateResidual
architecture and release architectural dimensions, but never load learned
release parameters, bounded7 parameters or cached body features. Architectural
variable/module assignments remain the same fixed schema.

Every fold uses its saved fit-only normalization and vocabulary support. Preserve
target/sibling masking. Fit/inner-selection/outer-test role assignments are fixed
from complete patient-hospital components. Only fit participates in gradients.
Inner selection is deliberately not used to pick epochs or hyperparameters.

AdamW lr1e-4, defaults otherwise, gradient clipping1, SmoothL1beta1, deterministic
seed19, four CPU threads, eight fixed epochs, evaluation/dropout behavior matching
the existing reconstruction training (`eval()` does not disable gradients).
Each epoch has1072 samples with replacement per target: equal exposure to the
selected/broad groups when both have fit support, otherwise the available group.
Use the existing bounded hospital sampling weights [.5,2]. No target removed by
held-out performance. Initial readout is zero-weight plus equal-group fit-label
weighted median in fold units. Both arms share initialization and sampled order.
EMA/future heads are not supervised in this current-value reconstruction task;
do not claim that the experiment retrains a future JEPA objective.

## Execution And Evidence

Execution can advance one complete epoch/arm/fold at a time, with hash-bound
resumable optimizer, RNG and model state. This does not shrink the five-fold,
two-arm, eight-epoch experiment. Intermediate fit loss is diagnostic only.
Do not evaluate/pick a checkpoint by outer outcomes before all fixed training
is complete. A single epoch is never sufficient to decide that new initialization
or physiological representation succeeds/fails.

After training, compare real/process/fit-population center on identical eligible
outer examples, convert predictions back to the original normalized units for
comparison, aggregate by patient and hospital/component and retain unsupported
or empty-history counts in coverage. Report every target, not only winners.
Fit-only populations are baselines; previous bounded7 scores are not independent
outer baselines because it saw these patients. Test folds overlap in their
training populations, so do not count model folds as independent replications.

No promotion follows without the full existing observation/placebo/external
requirements. A negative result at this fixed budget could still reflect
optimization limits; it does not prove a lack of physiology. A positive result
does not establish treatment causality or complete human-state identification.
