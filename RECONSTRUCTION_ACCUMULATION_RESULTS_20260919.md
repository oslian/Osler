# Balanced Accumulation Results

Two real-data candidate fits completed8epochs with identical expanded-source
sample plan and283008examples each. Balanced528-example accumulation gives536
AdamW steps perfit. SameLR selected8,sourceMAE1.878407; sqrtLR selected7,
sourceMAE1.817846. Source-only choice sqrtLR frozen before evaluation. Sequential
reference sourceMAE1.810509 remainsbetter. Four accumulation/scheduler tests pass;
explicit full-batch AdamW agrees with accumulated update to1e-12 on toy data.

| Reused cohort | Sequential expanded | Accumulated | Targets numerically better |
|---|---:|---:|---:|
| Broad | 1.269761 | 1.279049 | 9/33 |
| Selected | 2.911144 | 2.893760 | 16/33 |

Target-equal normalized MAE, not clinical units. Broadmacro worsens~.73%;selected
improves~.60%. Candidate vs sequential glucose improves inboth: broad.997046->
.980696;selected1.870194->1.823221, now numerically beating selectedpopulation
1.838811. BUN worsensboth(.893430->.898945; .863772->.903869). Creatinine harms
broad buthelpsselected. These means are not significance claims. No target passes
the reported corrected patient/site comparisons against sequentialexpanded,
small,old andpopulation. Matching accumulatedprocess/placebocontrols not yetfit;
cannotclaim fullpromotion regardless of this diagnostic gate.

The intervention supports a local glucose improvement but not a uniform repair.
Gradient averaging changes step count, clipping, Adam moments, effective decay
and learning rate together; cannot attribute gains solely to reduced variance.
This tests8patient/group-target accumulation, not the64patient localgradient
probe. Source-only checkpointselection still trades targets. Do not repeat this
exact accumulation experiment as untested or extrapolate to absent futurelabels.

Next isolate whether loss geometry/target weighting or optimizer exposure drives
the remaining tradeoff, using source-only development and fixed controls; do not
select per-target routes retrospectively from these evaluated cohorts. Original
33heads only;6untrainedstatus retained. No runtime, interval, registry or weights
promotion. This remains current-value masked reconstruction, not futureforecast.

Evidence:
- artifacts/native_reconstruction_accumulated_source_20260919.json
- artifacts/native_reconstruction_accumulated_evaluation_20260919.json
