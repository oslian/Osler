# Raw Observation Block Necessity Diagnostic

Question: which components support the raw HGB advantage in E17, before
choosing what a shared state should be trained to retain?
E3/E17 compare complete raw summaries against latent/process summaries.
E15 learns coordinates from body, not removal of particular observed blocks.
This diagnostic changes information, not readout architecture or encoder.

## Fixed Design

Use the exact E17 47 targets, five source folds, outcome-independent 512 fit /
256 selection patient limits and sample order. Retain target/history/sibling
masking from the original payload-verified cache. No outer queries or new
release predictions. MIMIC/NWICU roles remain reused development.

Original raw265 = latest53, mean53, presence53, lag53, logcount53.
Each arm stays 265-wide, with removed blocks replaced by constant zero in BOTH
fit and selection. Presence is always retained: this is not zero-as-missing.

- `latest_process`: zero mean; keep latest, presence, lag, logcount.
- `mean_process`: zero latest; keep mean, presence, lag, logcount.
- `values_presence`: zero lag/logcount; keep latest, mean, presence.
- `raw`: unchanged, refit to reproduce E17 raw predictions at atol 1e-12.

Use existing PARAMETERS HGB MAE100 iterations, seed19, learning rate.1,
15 leaves, min20, L2=1, no early stopping. Same bounded fit-component weights.
Three new heads plus one reproduction per cell: 940 fits maximum. Do not
tune parameters, sample membership or masking after observing results.
Re-training each ablation avoids testing only out-of-distribution corruption.
Constant padding equalizes schema, not all effective capacity or information.

## Evidence And Scoring

Bind protocol/code, original cache hashes, E17 reports/private predictions,
roles/scaling/provenance and library versions. Verify exact sample identities,
truth, components and saved E17 hashes; stop on mismatch. Keep predictions and
models private under ignored data; publish only aggregate results and hashes.

Seven fixed directional comparisons: raw minus each of the three ablations;
latest_process minus mean_process; each ablation minus E17 hgb_body.
Average repeated fold losses within patient first, then average by source
component. Per target report normalized MAE, paired 5000-resample bootstrap
95% CI and paired-unit one-sided t p; descriptive screen requires both levels'
CI upper < 0 and p < .05/(47*7), >=30 patients and >=3 components. These are
diagnostic screens, not release gates, nor proof of equality when nonsignificant.
Report all targets and complete family, not just beneficial examples.

## Interpretation And Stop

Loss after removing mean supports history-average utility for THIS fixed HGB;
loss after removing latest supports latest-value utility. Loss after removing
lag/count supports timing/count utility, which may be physiology or workflow.
These cannot prove state information loss, pure physiology or forecast gains.
If no block is reliably necessary, do not select an encoder repair from this
test. If a block matters, inspect its state retention under controlled readout
before any reconstruction-loss or encoder intervention. No automatic new run.
One fixed diagnostic; no parameter sweeps, promotion, runtime or interval edits.
