# Decay-Only Controlled Result

Eight epochs complete; same shared-state33heads, source queues/priors,528example
blocks, sqrtLR, betas, clipping and selection cohort. Only weight_decay changed
.01->.229721969.283008examples/536optimizersteps. Source selects epoch7 with
MAE1.820250 versus originalaccumulated1.817846. Both select7, reducing checkpoint
timing ambiguity for this comparison. No parameter selection on evaluation.

| Cohort | Original accumulated | Decay-matched |
|---|---:|---:|
| Broad | 1.279048825 | 1.280007590 |
| Selected | 2.893759874 | 2.907537184 |

Target-equal normalized MAE; broad+.075%,selected+.476%. This isolated increased
decay is not an aggregate accuracy repair. The existence of a decay-clock
difference is confirmed, but matching it is not supported as the missing
solution. No broader claim that all regularization choices are ineffective.

Both reused cohorts and exactsameprediction pairs as previous experiment.
No matched accumulated process/placebo controls, no untouchedexternal cohort,
no futureforecast or interval validation. Runtime, registry and release weights
unchanged.33trained/6pendingstatuses retained. Original evidence files preserved.
Do not repeat this exact decay matching as an untested repair; Adam moment clock
and optimizerstep count are separate unresolved factors, not automatically proven
causes. Parameter paths with skipped updates are not exactly decay-matched.

Evidence:
- artifacts/native_reconstruction_decay_matched_source_20260919.json
- artifacts/native_reconstruction_decay_evaluation_20260919.json
