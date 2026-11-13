# Native rate scaling audit

Read-only source-bound scaler audit on unique supported history tokens per role.
Hash-bound native cache/checkpoints verified. No fitting or outcome selection.
`artifacts/native_rate_scale_20260911.json`.

Source medians3, standard-deviation scales4.35-5.83. Normalized IQR ranges
.6006-.8542fit,.6006-.8730new source,.8580-1.1487hospital. Thus ordinary
observed rates do not numerically collapse to nearzero after normalization.
This is a numerical scale fact, not a learnability or utility guarantee.

Absolute normalized rate>5 occurs .32-.66%fit,.29-1.29%new source,.86-3.00%
hospital. Hospital standardized SD1.11-1.48 versus source1, so tails/support
do shift. No causation of forecast error or justification for posthoc clipping
has been established. Units/validity must not be inferred from these values.
The observed numeric input spans0-145 across the full token cache, but maximum
alone cannot identify invalid data or a principal failure mechanism.

This weakens simple scale-collapse explanations of tiny treatment increment.
Before adopting robust normalization, require source-only settings and a paired
real-label test; synthetic success and these marginal distributions alone are
insufficient. No runtime, training, interval or promotion changes.
