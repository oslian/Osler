# Exact Replay Learning Curves

Replayed the unchanged64fit/32tune seed7 pilot with all16 original tune scores
reproduced to1e-8. Added fitting MAE and retained final epoch8 checkpoints,
without changing training, sampling, model selection or original bound script.

| Arm | Fit initial -> epoch8 | Tune initial -> epoch8 |
|---|---|---|
| No cross-organ edges | 7.42657 ->7.37558 | 8.34714 ->8.41926 |
| Canonical sparse edges | 7.42657 ->7.37538 | 8.34714 ->8.41953 |

Fit MAE declines monotonically over the logged8epochs while tune increases.
This supports an in-sample/non-generalizing correction pattern in this small
pilot, not literal disconnected learning. It is not an exact full-loss analysis,
unique overfitting cause, convergence proof or population-wide impossibility.

Epoch8 is diagnostic only, not the source-selected model (which remainsepoch0).
New-source change versusbaseline +0.03585 CI[-0.08435,+0.15646] no-edge and
+0.03589 CI[-0.08438,+0.15660] sparse. Hospital -0.04298
CI[-0.14604,+0.05892] no-edge and -0.04290 CI[-0.14598,+0.05905] sparse.
No significant held-out gain or harm. Do not selectepoch8 because its hospital
point estimate is smaller, or present similar graph errors as equivalence proof.

There is no evidence here that adding the canonical graph fixes generalization.
Further work needs to establish useful state-dependent correction beyond anchor
features and adequate source support; extending epochs alone is not supported
by this pilot's tune trajectory. No settings were selected on held-out results.
No runtime/calibration/promotion changes. Final/selected checkpoints private.

Evidence: `artifacts/native_organ_replay_trace_20260913.json`.
