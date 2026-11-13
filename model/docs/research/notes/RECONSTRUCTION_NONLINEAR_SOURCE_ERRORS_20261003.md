# Where the New Readout Actually Loses

## Fixed Scope

All47 current-value masked reconstruction targets, all five existing source
selection folds; no outer rows, retraining, hyperparameter search or promotion.
The original patient/component aggregate MAEs reproduce for every arm. This is
not a future-forecast experiment. Source data have already been used for
development and these diagnostics are not independent confirmatory evidence.

For each patient, losses average across eligible folds before population
aggregation. Each occurrence receives inverse eligible-fold-count weight.
Truth bands use the absolute fit-normalized value: central <=1, intermediate
(1,3], tail >3. These are descriptive distances from the fitting reference,
NOT clinical normal ranges, stability classes or legal deployment selectors.
Different folds may assign the same measurement to different bands because
their fitting scalers differ. Fractional mass is therefore reported explicitly.

## Extreme Values Are Not the Main Source of These Regressions

Patient-equalized, normalized MAE. Negative contribution means benefit.

| Target | Native | Body probe | Central contribution to difference | Intermediate contribution | Tail contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferritin |0.827030|0.845038|+0.028461|-0.007874|-0.002579|
| Troponin I |5.184358|5.226455|+0.127101|-0.007243|-0.077760|
| O2sat |0.600807|0.602190|-0.000426|+0.001455|+0.000355|
| BUN |0.635639|0.614053|+0.007297|-0.015638|-0.013245|
| INR |0.893784|0.860433|-0.009932|-0.012387|-0.011033|

Ferritin's central band contains82.95%of weighted patient mass and87.10%of
summed positive body-versus-native regression. Troponin's central band contains
75.90%of mass and89.47%of positive regression. These percentages exclude
improvements in their denominator; they are NOT shares of total absolute error.
Both targets improve on average in their intermediate/tail bands. Removing
extreme values would therefore not repair the observed overall regression.

Troponin's central body error0.39764 exceeds native0.23019, process0.29548,
population0.13870 and matched shuffle0.26994. Yet its tail body error28.4992
improves on native28.9588. The readout trades better estimates away from the
fitting center for worse estimates near it. This is an error pattern, not proof
that the model can identify either group without knowing the hidden answer.

O2sat differs: central mean improves slightly; losses concentrate outside the
central band. BUN benefits outside the center enough to outweigh central harm;
INR improves in all three bands. One universal tail explanation is contradicted.
The current nonlinear learner already uses absolute-error loss; the old
squared-loss ridge tail failure is not a sufficient explanation of this result.

## Matched Local Bias Diagnostic

Within each fold/component, split patients by the existing fixed hash. Estimate
each arm's median residual in one half and apply it only to the other half.
Correct ALL SIX arms equally. Compare before/after on exactly the same supported
occurrences using the original patient weights. Insufficient halves remain
unknown, not zero correction disguised as evidence. This uses local labels and
is NOT a new-hospital calibration method or an enabled runtime correction.

| Target | Supported mass | Before: native / body | After: native / body |
| --- | ---: | ---: | ---: |
| Ferritin |99.60%|0.829559 /0.847388|0.826140 /0.838236|
| Troponin I |99.96%|5.186562 /5.228688|5.180178 /5.223657|
| O2sat |100%|0.600807 /0.602190|0.586984 /0.585547|
| BUN |100%|0.635639 /0.614053|0.633078 /0.612662|
| INR |99.99%|0.893806 /0.860463|0.870141 /0.852590|

Ferritin/troponin remain worse after matched centering. A constant component
bias is not a sufficient repair. O2sat's small mean ordering reverses, so a
calibration contribution is plausible there; this is neither a significant
benefit claim nor evidence of transferable correction. Across all47, nine
targets have uncorrected patient mean harm versus native;13 have harm after
matched offset. These sets differ and support can shrink: this is not a claim
that correction caused four additional failures on an identical full population.

## What This Changes

- Do not route by the observed target band: the hidden value is unavailable.
  The test does not demonstrate prospective stable/dynamic recognition.
- Do not delete tails or automatically switch every target to a robust loss.
  Ferritin/troponin regress in the center despite a robust objective.
- Do not treat a local offset as the general fix. Earlier affine/offset and
  site-constrained experiments already failed universal transfer; this matched
  diagnosis preserves those negative results.
- The next unresolved distinction is whether independent observation support
  and fit-to-query feature differences identify these adverse cases *before*
  revealing the target, versus the body representation/readout failing to
  preserve that information. Cached masked input summaries permit a source-only
  support diagnostic without a new architecture or another encoder extraction.
  No such support explanation is established by the present label-based bands.
- BUN/INR still warrant separate frozen-model component evidence, not the same
  repair as genuinely adverse targets. Their prior gate failures remain intact.

## Verification

`python -m scripts.diagnose_nonlinear_source_errors`

`python -m unittest tests.test_nonlinear_source_errors tests.test_nonlinear_source_components -v`

Seven tests pass: fold/patient weighting, exact contribution sums, matched bias
removal for all arms, unknown singleton/empty-band handling, rejection of outer
roles/duplicate pairs/nonfinite values, component checks. All47 completed with
source prediction hash and aggregate parity verification. Only aggregate
results are committed; no patient rows/identifiers. No model weights, runtime,
intervals, gates or promotion changed. The full goal remains unresolved.

Artifact: `artifacts/native_reconstruction_nonlinear_source_errors_20261003.json`.
