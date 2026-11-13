# Multi-Scale Masked MAP Reconstruction

Same922fit/334tune/311evaluation/194hospital patients and full target/sibling
masking. Added fixed1/6/24h summaries plus allhistory. Source tune selects
ridge alpha10000. Retrained allhistory incumbent reproduces previous MAEs
to1e-9. No significant advantage over population, incumbent or support.

Evaluation MAE17.29335 versus incumbent12.26777 and population12.00322.
Raw-incumbent delta+5.02559 CI[-.66457,15.76908]. Hospital raw14.13867 versus
incumbent14.27455; delta-.13587 CI[-.73879,.43349]. No promotion.

An exact retraining tail diagnostic reproduces all role results. Evaluation
median absolute error9.64772 but max1618.83663mmHg. One patient's error is
30.10% of total absolute error. The ten largest-error patients' mean absolute
linear feature contribution is dominated by CPK6h slope(144.71854mmHg), then
CPK6h std(13.85770) and CPK24h slope(3.14157). Hospital maxerror58.40322;
it does not show the same catastrophic tail.

This localizes the new multiwindow probe's extrapolation vulnerability to
short-window numeric features. It does not establish whether underlying
CPK events are erroneous, too closely spaced, or merely outside fit support.
Nor does it explain previous JEPA failures. Do not remove the affected patient
or tune clipping on heldout values. Next inspect source-fit temporal support
and numerical feature ranges, then validate any reliability rule source-only.

Evidence: artifacts/native_body_multiscale_raw_20260913.json and
artifacts/native_body_multiscale_tail_20260913.json. Only aggregate diagnostics
saved. No runtime weights, calibrator or registry changed.
