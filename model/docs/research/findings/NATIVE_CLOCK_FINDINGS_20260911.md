# Native time-encoding collision

Correction input now built: `artifacts/native_clock_tokens_20260911.json`.
36,192/128,236unique tokens change only columns159/171 (the two clocks).
Other data arrays and original cache remain unchanged. Three tests pass,
including negative pre-ICU offsets. The original clipping is two-sided [0,168]:
the aggregate collision count below must not all be attributed to the upper
bound alone. Both lower and upper saturation are removed in candidate inputs.
No accuracy retraining yet; no change to query context, labels, or serving.

Read-only audit of the hash-bound native schedule/token cache, with the same
active-step definition as the native filter (observed lab or known rate).
Aggregate: `artifacts/native_clock_20260911.json`.

Of263,668 adjacent active-history transitions,69,164 have positive physical
elapsed time but zero encoded elapsed time.3,824/5,526 queries contain at least
one such transition. Median lost interval is .516667 hours. These are query
history positions, not independent unique events or patients.

Code source: native_event_tokens.encode_group clips hours to[0,168] before
normalizing; NeuralBeliefFilter reads the last coordinate for elapsed decay.
A synthetic same-rate observation moved12hours later after168hours yields an
identical entire token. Thus the clock channel cannot distinguish that gap;
sequence order/update count still exist, so do not claim all timing is lost.

This is a reproduced representation limit, not proof of forecast degradation.
It was retained deliberately in the prior matched diagnostic, not silently
introduced by native conversion. Correction should preserve all other inputs,
query labels and roles, encode elapsed time without this saturation, then
retrain the same model and compare against the completed native baseline.
Do not replace serving clocks under incompatible weights/calibration or infer
accuracy gains from synthetic collision removal. No fitting/runtime/promotion.
