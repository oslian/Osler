# Constant Objective Diagnostic

Source-fit-only equal-group centers computed for all33joint-trained targets.
Weighted median minimizes equal-group absolute loss; SmoothL1 optimum uses
scipy.brentq on the clipped residual derivative(beta1 in normalizedunits).
Scored fit/encoder-selection separately; no external labels or neural fitting.
Three tests pass and allfit MAE optimality checks pass.

Encoder-selection macro Huber-minus-median MAE:
selected-.0107362,broad+.0110929; equal-group average+.0001784.
Differences are mixed, not a large uniform constant-bias explanation.
Glucose+.00683/+.01514;lipase-.03233/-.01900;CPK+.00144/+.02964;
troponin+.02690/+.17116;temperature+.00393/+.00524(selected/broad).
Sparse support remains explicit(lipase28/7,troponin34/18).

This does NOT rule out conditional neural loss effects; a constant predictor
cannot identify why an encoder/readout overfits. Earlier potassium L1 retraining
and magnesium loss-offset studies also did not yield stable repair. Do not
repeat loss replacement as an established solution or promote on these numbers.
The current evidence does not prioritize a universal loss-mismatch-only story.
Runtime unchanged; no candidateweights or thresholds modified.

Evidence:artifacts/native_reconstruction_joint_constant_objective_20260918.json.
