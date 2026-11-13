# Optimizer Exposure Clock Audit

Read actual end-of-epoch8 optimizer states for sequential and both accumulated
arms. Defaults verified from installed PyTorch2.12.0 signature and checkpoint
param_groups: AdamW betas(.9,.999),weight_decay.01. No optimization occurred.

| Arm | Maximum steps | First/second moment half-life in examples | Decay-only factor |
|---|---:|---|---:|
| Sequential | 283008 | 6.58 / 692.80 | .753514 |
| Accumulated sameLR | 536 | 3473.61 / 365798.69 | .999464 |
| Accumulated sqrtLR | 536 | 3473.61 / 365798.69 | .987759 |

Half-life is the exponential coefficient decay in observation-example units,
not exact effective memory after bias correction or patient independence.
Decay-only factor excludes gradient motion and must not be called observed
weight-norm shrinkage. Sequential81tensor states have283008steps,18have276357;
accumulated99statesall536.2212parameter tensors have no optimizer state in each
arm; unused/frozen model paths may explain that, and this audit does not establish
a disconnected-gradient bug. An initial overstrict diagnostic assertion that all
active tensors share the maximum count was corrected to report the distribution.

The prior accumulation experiment changes moment clocks and regularization in
addition to sample averaging, clipping, learning rate and update count. This is
a proven comparison confound, NOT a proven cause of its mixed prediction results.

For always-active parameters, decay alone can be matched exactly:
new_wd = (1-(1-reference_lr*reference_wd)^528)/new_lr.
At sqrtLR this gives.229721969; at sameLR5.278608964. A zero-gradient real AdamW
unit test verifies528sequential decay steps equal one adjusted step to1e-12.
This does not match parameters with skipped updates, or restore sequential Adam.
Matching coefficient half-lives by beta^528 yields(~6.92e-25,.589628), but averaging
then squaring gradients differs from averaging squared gradients. These are
diagnostic sensitivities, not a recommended automatic production configuration.

Next controlled intervention can hold accumulated sample order/rate fixed and
change only exposure-normalized decay; moment-clock adjustment must be separate.
No new accuracy, source selection, calibration, runtime or promotion result.
Evidence: artifacts/native_reconstruction_optimizer_clock_20260919.json.
