# Multi-Component Point-Offset Repair: Complete

## What Was Tested

All five frozen-model calibration predictions completed. Every role, raw label,
component identity and bound hash was checked. Fold2 used certified equivalent
fold0predictions; others were independently inferred with their own model.
Calibration-role patients/components and outer test patients/components are
disjoint WITHIN each fold, not globally independent across crossfit roles.

For each of47targets and5folds, each arm (physiology/process/population) receives
the same predeclared component-equal weighted-median offset procedure.189/235
target-folds meet the >=30patients in>=3components requirement;46remain exact
zero offset due insufficient support. No encoder weights or inputs changed.
All235offset records independently reproduced. For every supported arm the
weighted calibration absolute error did not increase, as required by the fit.

## Full Outer Results

All1,800,214outer pairs/73,670people retained.29/47patient-equal target means
improve over uncorrected physiology;27/47component-equal means improve.
Normalized target macro MAE:0.925567->0.917067(patient equal),
0.901157->0.897123(component equal). These are descriptive point estimates.

ZERO targets pass the full incremental screen: simultaneously beat the
uncorrected physiology AND corrected process/population at both aggregation
levels under p<.05/47. This is not zero signal or zero numerical improvement.

| Target | Raw patient MAE before | After | Remaining issue |
|---|---:|---:|---|
| MAP |13.26625|12.82157|Improves original at both adjusted levels, but component CI vs corrected controls crosses0|
| Heart rate |14.17640|13.94545|Beats corrected controls, incremental component p=.006843 misses .05/47|
| Glucose |37.78697|37.47480|Component uncertainty vs process, incremental multiplicity still fail|
| Creatinine |0.63340|0.61999|Incremental component CI[-.02187,.00787] crosses0|
| BUN |11.08154|10.97735|Incremental component CI[-.28812,.05852] crosses0|
| Sodium |3.63242|3.61534|Still worse than corrected controls at patient level|
| Bicarbonate |3.46311|3.45239|Component MAE worsens3.98593->4.03055|
| Albumin |0.38851|0.39832|Patient and component means worsen|
| Total protein |0.56045|0.56327|Patient mean worsens despite component improvement|

No cherry-picked target corrections are enabled. The previous13descriptive
reconstruction passes refer to uncorrected candidates; they were never runtime
promotions and are not silently replaced by this result.

## What This Localizes

MAP's original disadvantage is partly correctable additive bias: its patient
delta vs process changes from+.06147uncorrected to-.23776after equal calibration
of both arms. The incremental correction itself improves MAP significantly at
both prescribed aggregation levels. Thus the previous broad harm is not proof
that its latent contains no usable information. Yet component-level comparative
uncertainty remains; no universal transfer solution is established.

More calibration domains do not guarantee a universally transportable offset.
Albumin is a counterexample; sodium remains worse and bicarbonate aggregation
directions disagree. This rules out "only one global intercept was missing"
as a complete fix, not all forms of conditional modeling or calibration.
Clinical mechanisms, measurement differences, conditional response mismatch and
finite optimization remain unseparated. No unique root cause is claimed.

## Implementation and Limits

First scorer attempt stopped before fitting because it incorrectly enumerated
targets from fold0alone. A rare target can be absent in one fold. Fixed to the
five-fold union and added a regression test; this is a NEW scoring implementation
bug, not a cause of past model failures. No threshold/data/weight changed.

Five offset tests plus prior role/scoring tests cover weighted vs patient median,
lower ties, support, duplicate rejection, unchanged predictions and rare-target
union. Entire calibration outputs were independently replayed for offset/MAE
and hash invariants. No deployment, interval or forecast changes.

This follow-up was designed after outer outcomes were viewed. Outer remains
reused validation, not untouched confirmation. Overlapping fits and certified
reuse make bootstrap/t-tests conditional/descriptive. No90%interval calibration,
future forecast, clinical or causal claims. Candidate-only, no promotion.

Artifact: `artifacts/native_reconstruction_additional_calibration_20260925.json`.
