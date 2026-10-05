# Source Expansion Retention Localization

Both frozen selected real checkpoints replayed on the same original1252 source
patients. All132 encoder-selection target scores reproduce to1e-7. No fitting,
new checkpoint selection or new evaluation cohort. Original broad fit subset is
not the full expanded population; some of its pairs may not have been drawn
within the expanded4288-draw budget. Selected-source patients remain identical.

Expanded vs small pooled fit error worsens24/33selected and33/33originalbroad
targets; selection error worsens14/33selected and15/33broad. Fit harm alone is
not failed generalization: prior audit showed broad evaluation improves26/33,
consistent with less specialization to the old small sample, among other causes.

## Earliest Localized Glucose Failure

Selected glucose fit MAE1.130776small->1.213390expanded, compared with expanded
prior1.198364. The selected expanded model already loses the prior on fitting
patients(+.015025), not only unseen patients. Encoder-selection also worsens
1.017905->1.040399 and loses its unchanged prior1.026170. Therefore the latest
glucose failure cannot be described solely as fitting success followed by failed
generalization; that was accurate for an earlier joint checkpoint, not this one.

Expanded glucose selection history: selected-group epoch2=1.024504, versus
epoch7=1.040399; broad epoch2=.710012 versus epoch7=.774187. Both epoch2 values
beat their own initial priors, whereas the globally selected epoch7 loses both.
This is an observed target-retention tradeoff within source selection. It does
not authorize oracle target epochs, establish significance of tiny early gains,
prove conflicting gradients, or show that epoch2 would improve final evaluation.

## Other Targets Differ

Selected AST fit worsens+.133500 but selection improves-.325047; troponin fit
worsens+.236828 but selection improves-.790879, still losing population prior.
Their later evaluation harm cannot be equated with the glucose pattern. BUN fit
worsens+.009090 while selection improves-.031250 and later evaluation improves.
Do not impose a 'never increase fit MAE' criterion; that would reward overfitting.

Next training repair should test preservation of source-supported target gains
while retaining one shared state, using development information only. Compare
against the fixed expanded candidate and all existing controls. Do not assert
gradient conflict without a direct gradient/intervention test; different patient
exposure, stochastic schedules, priors and selected epochs remain entangled.
All33trained/6pending targets retained. No runtime, interval or promotion change.

Evidence: artifacts/native_reconstruction_exposure_retention_fit_20260919.json;
source-exposure training logs. Private rows remain in ignored data only.
