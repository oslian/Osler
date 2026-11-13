# BUN History Use: Site And Pre-Anchor Support

Fixed previous BUN full-history versus latest-per-variable predictions. No
training or inference rerun. Exact sample-level hospital mapping uses the
previously verified site manifest, including multi-site patients; no arbitrary
patient-level hospital assignment. Source pairs90selected/3086broad reproduce
the previous mean deltas. All original history lengths verified from shards.

Predeclared descriptive bins: independent event count1-8/9-32/33+, actual stream
span0-6h/6-24h/24h+, maximum age of each variable's latest real measurement
0-6h/6h+. Measurement age includes reporting delay and event-to-anchor gap.
No bins depend on future truth or residual direction. Groups overlap and their
95% paired intervals are unadjusted; cells below30patients have no interval.

## Hospital Results

Latest-per-variable minus full normalized reconstruction MAE:

| Site | Selected N | Selected change | Broad N | Broad change |
|---|---:|---:|---:|---:|
|422|4|-.04793|226|-.01348|
|435|2|+.01880|1102|-.01630|
|445|13|+.08089|227|-.04421|
|449|71|+.05337|1531|-.05188|

All4 broad site means favor latest coverage; three broad intervals exclude
zero,422 crosses zero. Equal-site means remain opposite: selected+.02628,
broad-.03147. The broad result is not produced solely by patient weighting
or a single outlying hospital. Selected has adequate per-site support only449,
whose interval[-.01010,+.11665] crosses zero; do not claim four-site selected
replication or definitive cross-population heterogeneity from sign alone.

## History Support

Broad change by event count:1-8-.00143(N705, CI crosses0),9-32-.03184(N1077),
33+-.05765(N1304). By span:0-6h-.00079(N594, CI crosses0),6-24h-.00599(N762),
24h+-.06095(N1730). This localizes most benefit to longer repeated histories.
Both fresh maximum-latest-age and older strata improve in broad: -.02724 and
-.04124. Simply excluding variables whose latest value is older than6h does
not explain away the broad result.

Selected33+events changes+.07306(N62, borderline nominal positive interval).
Within the same449site and33+event stratum: selected+.06420(N59,
CI[-.00810,+.13519]), broad-.06274(N831, CI[-.08008,-.04553]). Sign reversal
persists descriptively after these coarse restrictions. Neither hospital alone
nor a coarse event-count threshold identifies a universal successful route.

## Decision

Do not deploy site-based or length-only truncation. This audit narrows the
problem to how the fixed state uses repeated history in different contexts;
case mix, values/changes, acquisition patterns, initialization and transition
drift remain unresolved. A controlled separation of history content from the
elapsed-time transition path is distinct from the prior final-update or suffix
experiments. Such a test must preserve latest coverage and the original anchor,
and cannot be presented as physiological causal evidence.

Three age/bin tests pass. No private rows written to public artifacts; only
aggregate site/stratum results. No model, runtime, calibration or promotion
changed. Evidence: artifacts/native_reconstruction_bun_history_strata_20260920.json.
