# BUN Dynamics: Fitting Loss And Actual Exposure

Fixed bounded7; all529selected and9664broad available fitting BUN pairs.
Normal versus body-dynamics-off, same observations and checkpoint. Evaluate
both MAE and exact PyTorch default SmoothL1(beta1) used in training. No fitting.

| Group | Full MAE | Body-off MAE | Full SmoothL1 | Body-off SmoothL1 |
|---|---:|---:|---:|---:|
| Selected529 |.854718|.839831|.492712|.485062|
| Broad9664 |.936551|.906467|.552663|.532080|

Available fitting pool is not identical to actual sampled exposure. Therefore
replay all8epochs of the exact bounded sampler, verifying each epoch's target
counts, final RNG state and full saved seen-patient sets. Extract seen BUN pairs
at selectedepoch7, never use the finalepoch8set for that comparison.

| Group/scope | N | Body-off MAE change | Body-off SmoothL1 change |
|---|---:|---:|---:|
| Selected seen |527|-.014920|-.007672|
| Selected unseen BUN |2|insufficient|insufficient|
| Broad seen |2934|-.021316|-.014481|
| Broad unseen BUN |6730|-.033906|-.023243|

Broad seen95%paired intervals: MAE[-.026240,-.016341], SmoothL1
[-.018486,-.010482]. Selected seen SmoothL1interval crosses zero. Unseen BUN
does not mean unseen patient: other targets may have used those patients.
These are unadjusted in-sample descriptive intervals, not promotion tests.

## Localization

The broad BUN body-transition penalty already exists on actual sampled training
pairs under the actual per-example training loss, not just on new patients or
because MAE differs from SmoothL1. This excludes external shift and evaluation
metric mismatch as complete explanations of this particular BUN penalty.

It does not prove the optimizer is converged, that all per-target gradients
conflict, or that BUN-only loss represents the shared training objective. The
checkpoint is jointly trained and selected on a macro score across targets;
other targets benefit from body dynamics. Fit-patient-equal loss here is not
the historical draw-weighted objective or site-weighted loss. Earlier small
gradient-conflict and global-dynamics-deletion experiments must not be repeated
as if untested.

Next investigate whether a source-fit constrained update can lower the BUN
training contribution without harming other target losses, with matched
patient support and separate selection. Do not deploy body-off based on these
results, and do not claim new external validation. All33trained plus6pending
scope remains; no future-forecast repair inferred from current reconstruction.

Four loss/hook tests pass, including equality to the library's SmoothL1 and
hook cleanup. Exact sampler assertions pass. No runtime weights, intervals,
registry or promotion changed. Evidence:
artifacts/native_reconstruction_bun_dynamics_fit_20260920.json and
artifacts/native_reconstruction_bun_fit_exposure_20260920.json.
