# Robust Matched Raw/Body Probe Protocol

Same current masked history/state caches, all33targets, source-only64dim
projections and weighting as prior ridge probe. Use verified exact-constant
handling. Replace half squared error with SmoothL1beta1; preserve equivalent
L2alpha100/n on coefficients and no intercept penalty. No original-head
anchoring or per-target tuning. L-BFGS-B max2000/ftol1e-12/gtol1e-7; require
solver success and finite coefficients. Same predictor dimension in all arms.

Compare raw against body, process and original using existing patient/site
incremental gate. Retain all targets and extreme labels. No waveform/clinical/
causal claim, fresh lockbox claim, or promotion on a probe result alone.
Purpose: remove squared-loss tail preference as a confound in comparing
representations. PCA truncation and missing temporal detail remain limitations.
