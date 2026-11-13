# Robust Dimension-Matched Raw/Body Probe Results

All33targets, exactly the previous current raw/body inputs, fit-only64dim
projections and weights. Replace half squared loss with SmoothL1beta1 while
retaining equivalent L2alpha100/n coefficient penalty and unpenalized intercept.
Use the verified exact-constant numerical repair. All99fits converge; no
target omission, outlier deletion, hyperparameter search or encoder update.

| Normalized macro MAE | Original | SmoothL1 body | SmoothL1 raw | SmoothL1 process |
|---|---:|---:|---:|---:|
| Selected |1.797984|1.850957|1.901196|2.054017|
| Broad |1.559771|1.574443|1.610748|1.716253|

Previous squared-loss body was3.197392/2.708317, raw3.237073/2.668227.
Thus the poor probe macro substantially depends on the loss, not solely body
information retention. CPK body becomes7.60376/3.09125 versus squared-loss
21.75588/18.24512. Robust fitting is not a universal gain: PTT selected body
2.19536 is worse than squared-loss1.90858. Do not choose per-target losses
after these outcomes and call that independently validated.

Raw beats body numerically14/33selected and13/33broad; zero raw targets pass
the existing patient/site gate against body, process and original. Overall
body still outperforms raw summary in bothgroups but loses original macro.
This is a diagnostic readout, not a replacement candidate promoted by average.

## BUN: Why The Apparent Raw Advantage Is Not Enough

BUN MAE selected original1.10547/body1.05789/raw1.03670, broad.89361/.81821/.79539.
Raw-minus-body patient means are-.02120/-.02282, but equal-site means are
+.02447/+.00570. Selected patientp.3568; broadpatientp.0117 is above the
prespecified.05/39 threshold; both sitep>.5. Gains do not distribute evenly
across sites. This is not merely 'almost passed' or a basis for lowering gates.

Glucose and PTT body beat raw in bothgroups, including after loss alignment.
Therefore this raw summary does not establish a broad encoder-discarded-signal
cause. The test remains limited by source-only unsupervised PCA, compressed
history summaries and shared fixed linear readout; it does not prove raw event
sequences contain no additional signal or establish end-to-end independence.

Two tests pass: in the quadratic region the solver reproduces the matched
Ridge objective, and one extreme label does not set a constant predictor to
the sample mean. Full audit exit0, no scaler warnings. No runtime, registry,
intervals, causal permissions or promotion changes. Reused source development;
masked current reconstruction only. No further readout sweep justified as a
new sole-cause remedy by these results.

Evidence: artifacts/native_reconstruction_robust_raw_body_probe_20260920.json.
