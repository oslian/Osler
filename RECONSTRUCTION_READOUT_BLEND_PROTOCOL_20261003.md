# Same-State Readout Complementarity and Component-Held-Out Blend

This is NOT a new ensemble method. Prior near-miss portfolios (20260908) and
clean own/full glucose blends (20260911) failed on different prediction sets.
The present question concerns the new frozen current-reconstruction body HGB
versus the original readout of the SAME trained state, all47 targets/all5folds.
No alternate body model, raw-feature deployment, new encoder or outer labels.

First quantify patient-equal and component-equal signed-error correlations,
opposite-error fraction and the exact cancellation obtained by a fixed50:50
blend. Cancellation relative to AVERAGE member loss is not gain over the better
member, which must be compared separately. These are retrospective diagnostics,
not a per-row outcome oracle or legal routing features.

Intervention: convex point = native + weight*(readout-native). For each target
and original source-selection fold, hold out one entire component. Fit the ONE
constant weight on remaining source components by equal-component MAE. The
exact solution uses the weighted median of (truth-native)/(readout-native),
weighted by absolute prediction difference and inverse component size; clip the
coefficient to[0,1]. Zero-difference rows are constant in the objective. Use the
existing weighted_median helper, no new optimizer/library or prediction clipping.
Ties use its lower weighted median convention. If all proposals agree, weight0.

Apply identical fitting to native+body, native+process and native+trained-shuffle.
Require30 training people and2 other components. Unsupported groups retain
explicit status and native-only bookkeeping and cannot pass. Never drop groups.
Do NOT blend the previously selected prefix predictions: their selection for
other components can have consumed the currently held component's outcomes,
which would leak through another layer of fitting. Use only fixed100-stage
predictions whose encoder/head fitting excludes all current source components.

Primary body blend comparisons: native, body100, population, fixed50:50 blend,
process blend and shuffled blend. All six contrasts share .05/(47*6) one-sided
t descriptive screening at BOTH patient and component levels, conditional
paired-bootstrap95% upper bounds below0,30patients/3components and complete
support. Average each patient's fold LOSSES, not repeated rows/predictions.
Uncertainty is conditional on overlapping existing fits/selection; it does not
include refitting weights in each resample and cannot authorize promotion.

No global weight may be selected from the outer cohort or from the scored
component. No target excluded because it failed earlier. Report all47 means,
all controls, weight distribution, unsupported cases and per-component losses.
Saved private predictions must reproduce original source support and metrics.
Only aggregate artifacts are versioned; no weights, registry, intervals or
serving behavior change. New lockbox/full runtime gates remain separate.
