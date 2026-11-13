# Event-window history diagnosis

Fixed-grid snapshots can omit observations between lags. Add pre-anchor6/24h
event summaries to the same original potassium6174pair source cohort. Use
latest known source versions at the query, not final future revisions. Numeric
features are standardized mean/std/min/max/last distinct-time change/slope;
process features are log event count/time span/slope-valid mask. All observations
must be known by anchor. No treatment, future label or new patient inputs.

Same source fit/tune/evaluation partitions and source-supported variable mask,
same source-selected tree settings as the original raw-history comparison.
21fits own/full/support. Support retains own numeric history and other variables'
observation summaries but removes their numeric summaries. Old model hashes
and source evaluation scores must reproduce. No new hyperparameter selection.
Compare augmented models with original models on identical evaluation patients,
also full-own/support/persistence. Patient-equal MAE and1000patient bootstrap.

Quantify whether event extrema extend beyond represented lag extrema on
query-variable-window entries with both available. This is descriptive input
loss, not proof those events predict future truth. Existing observation at
lags uses <=6h age; event summaries retain measurements within their window.
This comparison does not isolate age truncation versus temporal downsampling.
Identical tree counts do not imply equal effective capacity after adding inputs.

Source-only, reused development diagnostic; no external read, calibration,
runtime, promotion or causal claim. Passing source checks would justify a
frozen transfer test, not adopting a new production architecture.
