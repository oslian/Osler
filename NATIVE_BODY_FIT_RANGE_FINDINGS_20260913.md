# Sparse Slope Support And Extrapolation Repair

The 6h CPK slope is supported by only2 of922 fitting patients, and none of334
tuning patients have two CPK events in that window. Fit event spans range
1.53-4.32h. The evaluation role has one such window with a much shorter span
(about13min); its absolute slope exceeds the largest fit slope by about675x.
This explains why strong ridge regularization alone failed to protect the
new multi-scale probe from an extreme unseen feature. It does not prove the
underlying lab values are wrong; availability-time slopes are not necessarily
biological rates. This is a local finding, not all historical JEPA root causes.

Uniform fit-only numerical-summary min/max bounding was fixed before running
the new comparison, applied to all numeric summaries, not just CPK. All patients
and labels remain. Source-tune selects alpha, no heldout cutoffs were used.

| Role | Unbounded MAE | Bounded MAE | Population MAE |
|---|---:|---:|---:|
| Evaluation311 |17.29335|11.91145|12.00322|
| Hospital194 |14.13867|14.09913|14.30928|

Evaluation paired bounded-unbounded CI[-15.92891,-.05301]; maxerror falls
1618.84 to72.00mmHg. Hospital difference NS. Bounded vs population and
bounded placebo remains NS in both roles. Therefore catastrophic-extrapolation
mitigation is demonstrated on reused validation, but useful cross-variable
MAP reconstruction/promotion is still not established.

Tradeoff: feature bounding suppresses genuinely novel extreme physiology and
does not supply calibrated intervals. No raw observations are overwritten.
Fresh independent validation would be needed for any generalization claim.

Evidence: `artifacts/native_body_fit_range_20260913.json` with script/protocol/
cache bindings. Candidate-only numerical probe; no release, model weights,
registry or calibration changed.
