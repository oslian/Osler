# Prediction-State Residual Shift

Diagnostic only; no new calibrator fitted. Source prediction tertiles define
fixed low/middle/high bins, then both source-calibration and outer predictions
use those same boundaries. True values do not define bins. Each hospital
component has equal total weight in each dataset. Quantile boundaries are
source-patient quantiles, not component-weighted quantiles.

Decompose the difference in MEAN residual (truth minus prediction) exactly:
composition = sum((outer_mass-source_mass)*average_bin_residual),
within = sum(average_mass*(outer_bin_residual-source_bin_residual)).
Their sum must equal the observed overall mean-residual difference. This is
NOT a median-offset/MAE decomposition, a causal explanation or uncertainty test.

235target-folds examined;229have all bins observed on both sides.139also have
>=30patients in>=3components in EVERY bin on both sides. Among these139,
124have larger absolute within-bin contribution than composition contribution.
Do not infer139independent replications or sampling-noise exclusion.

Examples: albuminfold2mean-residual shift+.16055 comprises+.00519composition
and+.15536within-bin shift. Its high-prediction bin mean residual changes
-.212to+.088. MAPfold4 shift+6.65577 comprises-.01859composition and+6.67436
within-bin. O2satfold3 shift+1.03252 comprises+.06444composition and+.96808within.
The latter was one of the source-selected offset failures.

This weakens the simple hypothesis that transfer failure is explained only by
different proportions of low/mid/high predictions. It does NOT prove genuine
physiological conditional shift: bins are coarse and endogenous model outputs,
and patient state, measurement processes, mix WITHIN bins and sampling variation
remain unresolved. No per-bin correction is justified by these outer signs.

Next discriminating analysis should use pre-anchor observable state/support
matched within source-defined prediction bins, rather than adding more heads
or selecting offsets by external labels. Still candidate-only, no runtime or
promotion/forecast/interval changes.

Three tests verify decomposition identity, target-truth-independent cutoffs
and no invented empty-bin means. A new pandas read-only normalization error
was caught before results and repaired without changing weights or data.
Artifact: `artifacts/native_reconstruction_conditional_offset_shift_20260925.json`.
