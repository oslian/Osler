# Frozen Body Readout Diagnostic

Use both selected multi-target sampling checkpoints and identical patient-target
examples. Capture body state immediately before the existing readout. Per target,
fit StandardScaler+Ridge(alpha1/10/100/1000/10000) on fit patients; select against
fit-population median using tune MAE. Train identical same-target cyclically
deranged-patient placebo, independently within each role. No self donors in fit
or tune since eligible target support>=5/3. Cases<2heldout patients unscored.
Original prediction MAEs must reproduce. Compare real probe vs original,
population and placebo with existing patient bootstrap. No encoder update,
no candidate selection on heldout. Small reused validation; no promotion.
Encoder was trained on same fit patients, so this does not isolate unbiased
stacking or show latent differences are causal physiological mechanisms.
