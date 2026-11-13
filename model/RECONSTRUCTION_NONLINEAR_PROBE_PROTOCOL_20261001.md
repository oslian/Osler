# Frozen-State Nonlinear Information Probe

## Question

Can the current epoch-10 body state support a more accurate nonlinear readout,
and does the unprojected masked raw summary add information not recovered by
the same learner from that state? Earlier 33-target raw/body probes used
PCA64 and linear Ridge/SmoothL1 heads. They did not show a transferable raw
advantage and remain valid negatives, but cannot settle this different probe.

This is a diagnostic family extension, not evidence that more heads or a new
architecture are inherently better. No encoder weights are changed.

## Fixed Scope and Roles

All 47 eligible targets and five existing folds. Frozen epoch-10 real model,
same full native histories, target-and-sibling masking, actual times, fit-only
scalers and source query identities. Extract BOTH fitting and source-selection
roles. Never use outer, calibration or selection labels to fit preprocessing,
heads, thresholds, model selection or stopping. No posthoc harmful-group-only
training. Fold2 fit0 reuse requires an explicit exact-equivalence certificate;
its different selection queries still require extraction.

The source roles are reused development, not a pristine lockbox. Fits and
selection populations overlap across folds, not independent replications.

## Fixed Comparison

- Unchanged native body readout.
- Body64 -> HistGradientBoostingRegressor.
- Raw265 -> the same HistGradientBoostingRegressor.
- Process265 (same raw summary with latest/mean numeric values zeroed) ->
  the same HistGradientBoostingRegressor.
- Existing fit population reference.

Use the installed scikit-learn, loss=`absolute_error`, max_iter=100,
learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20,
l2_regularization=1.0, early_stopping=False, random_state=19. No PCA, feature
selection, depth/loss/rate sweep, or source-selection choice. Same bounded
fit-component weights as the existing trainer. No histogram cut points are
fitted on query rows. Sparse fit support must remain explicitly recorded; it
must not silently remove targets from the planned family.

This matches learner settings/tree leaf budget, NOT representation dimension
or exact effective capacity: raw has 265 columns, body64, process has fewer
nonconstant columns. Raw summaries still omit detailed sequence ordering. A
failure cannot prove raw information absence; a gain does not by itself prove
the encoder discarded it.

## Evidence Required

First verify complete feature grids, parent hashes, disjoint fit/query patients,
model identity, mask semantics, and exact unchanged-readout reproduction on
every source-selection query. Extract body via readout pre-hook without
changing the model. Features and patient identities stay in ignored local
SQLite caches with checksummed payloads and atomic committed progress.

Report all targets under patient and component weighting, matched comparisons
against original, body, process and population. Use existing fixed 47-family
patient/component paired summaries, not a lowered gate. Do not pool overlapping
folds as independent patients. This source-only stage cannot promote anything.
Only after the complete fixed source comparison may the exact frozen procedure
be assessed on existing outer data as reused development. A genuine final
external claim still needs new untouched data.

No forecasting, interval, clinical or causal claims; runtime and release stay
unchanged. Data preparation completion is not predictive efficacy.
