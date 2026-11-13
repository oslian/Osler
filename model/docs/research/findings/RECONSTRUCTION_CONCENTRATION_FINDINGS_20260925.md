# Source Correction Benefit Concentration

Reused all 162 matched point-only versus full-Huber source-CV target-folds.
No refit, outer access, new prediction head or threshold selection.

54 mean improvements versus unchanged become 43 when requiring a majority
of components to improve, and 36 when the mean remains negative after removing
any single component. Only heart_rate fold0 improves every component.
These are descriptive influence checks, not additional promotion gates.

Across all162, excluding query components with fewer than30 people changes the
sign of the mean in30. This is sensitivity evidence, NOT permission to exclude
these patients or replace the original population. Equal-component weighting
protects smaller domains but makes their noisy means consequential.

MAP improves the unchanged mean in allfivefolds, improves a majority in each,
and retains improvement after any one component is removed. Its supported-only
means remain negative, though fold3 is only -0.004414 rawunits. Chloride's fifth
fold improves the original mean by -0.014260 but lacks majority/leave-one-out
stability; supported-only mean is +0.018044. Thus the prior fivefold average
statement should not be interpreted as uniformly robust benefit.

No target previously beat all comparators across fivefolds. This audit does not
change that decision. It identifies benefit concentration and limited support
as contributors to unstable summaries, not the sole cause of model failure.
MAP is a potentially reproducible calibration effect versus unchanged, not
proof of added cross-organ information or future predictive ability.

Two tests passed. Runtime, encoder, intervals and promotion unchanged.
Artifact: artifacts/native_reconstruction_concentration_20260925.json.
