# E39: exact E38 optimizer-stage accounting

Question: where does local per-example gradient protection cease to hold on
the SAME update? E37 decomposed origins at the final parameter ray; E38 changed
training gradients. Neither separated momentum, preconditioning and decay on
each E38 step. Older moment-clock/decay trials are negative interventions on
different accumulated models, not this diagnostic.

Replay BOTH original E38 arms, all five folds, all 2048 examples/two epochs.
Use the unchanged installed torch.optim.AdamW and original combine/helper APIs.
Only a new diagnostic driver mirrors the bound training loop; never edit the
hash-bound trainer. Every original state tensor and epoch log must reproduce
exactly before any result is interpreted. Same-target pairs remain unprojected.
No alternative optimizer, new hyperparameters, selected targets or new weights
for serving. Original E38 query results remain authoritative because the final
models must be identical; no claim of a new efficacy trial.

For each current example gradient g, inspect these vectors in PARAMETER-STEP
units using actual post-step optimizer moments m,v and step t:

1. proposal = -lr*q, with q the actual clipped combined gradient.
2. momentum = -lr*m/(1-beta1**t).
3. preconditioned = momentum/(sqrt(v/(1-beta2**t))+eps).
4. with_decay = preconditioned - lr*weight_decay*theta_before.
5. actual = theta_after - theta_before.

Stages 1-4 are algebraic accounting, not literal torch execution order (torch
applies decoupled decay before addcdiv). Actual minus with_decay is the residual
of float32 arithmetic relative to these double-precision equations, not all
machine error in the training process. Record its maximum coordinate size.
No state/moment mutation. Reject nonfinite/invalid moments or optimizer variants.

Use the SAME threshold 1e-12*(1+||g||*||step||) at each stage; retain dots and
thresholds. Separately reproduce E38's UNSCALED proposal counts/logs exactly.
The scaled proposal count need not equal E38's count because of the absolute
tolerance; never attribute that difference to optimizer damage.

Record all 40960 example-step checks, including fold/arm/epoch/step/slot/target,
stage harmful flags, safe-to-harm/harm-to-safe flips, first harmful stage, and
the stages of FIRST loss of protection among actual-harmful examples. Count
same-target and cross-target pairs separately. These are repeated training
observations, not independent patients or statistical generalization evidence.

After each actual step replay the SAME two examples (no grad), recording
float32 predictions, SmoothL1 losses and absolute errors before/after. Compare
first-order actual dot with finite loss sign using tolerance
1e-12*(1+abs(before_loss)+abs(after_loss)); report ambiguous/ties separately.
Signed MAE and SmoothL1 changes are local and not held-out evaluation. Reuse
original float64 times without conversion.

Tests: momentum flip, positive diagonal scaling flip for an individual gradient,
decay flip, finite/shape/moment validation, real AdamW formula/rounding, zero
gradient, and exact miniature E38 model/log reproduction for both arms.
Independent scalar row aggregation and transition counting must agree with
vector summaries; original bindings rechecked at end, full target inventory
retained. Save per-step rows privately, aggregate only publicly. Preserve10GiB.

Success is identifying local stages with reproducible accounting, NOT fixing
the model. If actual steps are harmful despite no earlier harms, investigate
numerics; if primarily after momentum or preconditioning, the next intervention
must address the actual step, not merely repeat raw-gradient projection. If
local gradients/finite losses disagree substantially, do not treat first-order
protection as a finite-step guarantee. Even successful local protection cannot
prove unseen-patient transfer. No runtime, interval, forecast or causal claim.
