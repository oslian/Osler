# Source-CV Anchored SmoothL1 Readout

Same current bounded7body features and linear head capacity. Add a quadratic
penalty around the original head in source-training-standardized coordinates,
including intercept. Fixed strengths.01,.1,1,10 plus unchanged. Select by
5-fold GroupKFold on source-fit hospitals only, minimizing group-equal then
site-equal MAE; fitting uses the same expected bounded-site weights. Scaling
and correction fit independently inside each fold. Original encoder already
saw fitting sites: inner CV assesses readout correction, not independent
end-to-end validation. Final coefficients frozen before source-selection scoring.

Selections:9unchanged,9strength.1,7strength1,5strength10,3strength.01. All33
targets included, source-only selection, no rank/strength choice from held-site
outcomes. Hash-bound body feature cache identical to the SVD diagnostic.

| Source selection | Original | Unregularized refit | Anchored refit |
|---|---:|---:|---:|
| Selected90 |1.797984335|1.896873820|1.817771650|
| Broad3323 |1.559770689|1.601832406|1.556646997|

Anchoring substantially reduces the previous regression. Broad now improves
about.20%on average; selected still worsens1.10%. Existing incremental
patient/site39-adjusted gate: selectednone,broadBUNonly,intersectionzero.
This does not establish a shared two-group repair, and no candidate is promoted.

## Interpretation

Weakly constrained same-capacity readout refitting was one contributor to
selection regression; bounding departures helps. Remaining cross-group gap is
not solved by this regularization. This is not proof the encoder lacks all
useful information, nor permission to tune more penalties on these outcomes.
The source CV is conditional on a pretrained encoder and its selected checkpoint;
future representation selection must respect that dependency explicitly.

Two tests verify large-penalty anchoring and equal-group/site scoring. Allfits
converged; auditexit0. No release weights, registry, intervals or runtime
changes. Private candidate only, current masked reconstruction not forecasting.
Evidence: artifacts/native_reconstruction_anchored_readout_20260920.json.
