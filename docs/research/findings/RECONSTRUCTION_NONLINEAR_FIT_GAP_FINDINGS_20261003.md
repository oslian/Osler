# Frozen Nonlinear Readouts: Where the Advantage First Disappears

## Question and Evidence

Does the new body-state readout fail to fit its own population, or does its
advantage disappear on source-selection groups that did not participate in fit?
This completes `RECONSTRUCTION_NONLINEAR_FIT_GAP_PROTOCOL_20261003.md`, not a new
model search. All 47 targets and five original folds are complete: 3,521,860
fitting patient-target occurrences, frozen encoder and saved 100-iteration HGB
readouts. Repeated people and equivalent fits are not independent replications.
No outer outcomes were read and no weights were refitted.

Report: `artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json`.
All 25 bound files match their hashes. Fit cache, original baseline, model
capacity, patient/component separation and original selection metrics were
verified. Fourteen focused fit-gap/support-selector/error-band tests pass.

## Overall Fit Is Usually Not the First Failure

Against the native readout, body HGB has these descriptive sign patterns across
235 target-fold comparisons. These are counts, not significance tests.

| Fit -> selection relative benefit | Patient equal | Component equal | Bounded weights |
|---|---:|---:|---:|
| Benefit -> benefit | 178 | 174 | 172 |
| Benefit -> nonbenefit | 54 | 58 | 60 |
| Nonbenefit -> benefit | 2 | 2 | 2 |
| Nonbenefit -> nonbenefit | 1 | 1 | 1 |

The three overall fit nonbenefits against native are myoglobin folds 0, 1 and 2.
Body improves the population and process-summary controls on fit in all 235
comparisons. On the actual bounded-weight training objective it also beats both
the weighted-median constant and the shuffled model evaluated on its original
donor-state training assignment in all 235 comparisons. The report separately
records the shuffled head evaluated on real individual states; that is NOT the
input assignment on which the null was trained.

Thus inability to fit, missing optimization altogether, or a selector that
never opens cannot explain most of these new readout regressions. This does
not prove the optimizer is ideal or that the shared representation is sufficient.

## Two Different Failure Locations

These are fold-equal descriptive means of patient-level normalized MAE, not
physical units or the earlier patient-collapsed pooled estimates. Compare native
versus body WITHIN each role; direct fit/selection gaps mix case mix.

| Target | Fit native | Fit body | Selection native | Selection body |
|---|---:|---:|---:|---:|
| Ferritin | 1.194067 | 1.020244 | 0.754141 | 0.770918 |
| Troponin I | 4.905960 | 4.659291 | 5.032937 | 5.082174 |
| O2 saturation | 0.583833 | 0.544681 | 0.601839 | 0.602515 |
| BUN | 0.649897 | 0.584871 | 0.636389 | 0.615073 |
| INR | 0.825496 | 0.759528 | 0.901565 | 0.870072 |

Ferritin and troponin I are the only targets with overall body-vs-native fit
benefit but selection harm in all five folds. Their conditional behavior differs:

| Central band only | Fit native | Fit body | Selection native | Selection body |
|---|---:|---:|---:|---:|
| Ferritin | 0.305633 | 0.176023 | 0.341006 | 0.375741 |
| Troponin I | 0.222717 | 0.272556 | 0.228050 | 0.401309 |

Central means absolute fit-normalized truth <= 1, NOT clinical normality,
stability or a pre-anchor feature. Truth-defined bands cannot route deployment.

- Ferritin: central fit error improves in every fold, then central selection
  error worsens in every fold. The first observed reversal is transport from
  fitting to unseen source components. Overfitting, component distribution
  differences and observational support remain competing explanations.
- Troponin I: central error ALREADY worsens during fitting in every fold,
  despite better overall fit MAE. Improvement elsewhere compensates for this
  conditional loss; selection adds a larger central regression. This is not
  purely a generalization problem or evidence of an optimizer bug. An aggregate
  absolute-error objective can prefer this tradeoff.
- O2 saturation: overall selection harm occurs in only two of five patient
  means, with a small macro reversal. Do not describe it as universal failure.
- BUN and INR: overall fit and selection improve in all five folds. Earlier
  corrected component-confidence blockers remain; positive means are not
  promotion and this diagnostic does not remove those blockers.

## Representation Precision Is Not the Explanation

The first attempt compared medians from two precision conventions and stopped
before completing a target. Original baseline centers are float64; cached
native labels passed through float32 before normalization. The corrected
diagnostic preserves the original float64 baseline and verifies every cached
label against the exact float32 conversion convention. All 235 comparisons
pass; maximum normalized conversion difference is 2.7177918582310667e-05.
No tolerance was loosened on the original selection baseline and no prior
experiment was silently changed. This bookkeeping correction does not account
for the observed prediction regressions.

## Decision and Remaining Repair Question

Confirmed: most overall adverse new-head results are not failure to improve
training fit; troponin also has a conditional tradeoff already inside fitting.
Not established: one universal root cause, absent physiological information,
or a repair that generalizes. More capacity or a new support-only selector is
not justified by these observations.

A discriminating next intervention is source-component-held-out selection of
prefix complexity for the SAME fitted boosted readouts, with native and matched
shuffle controls. This would test whether extra boosting stages cause harmful
specialization without changing the encoder. It remains unexecuted here and
must not use outer labels or truth-band oracle routing. Reduced complexity can
also remove real subgroup gains; positive average changes alone are insufficient.

Troponin's conditional fit tradeoff needs a separate diagnostic: determine
whether harm appears early or only after later stages, before proposing another
loss or conditional model. Do not infer that amplifying outputs, clipping values
or tuning a global offset fixes it; prior matched-offset tests already failed
for these targets.

This is masked CURRENT reconstruction, not future forecasting or evidence of
clinical/causal capability. Source-selection data have already been reused.
No runtime registry, model weights, intervals, gates or promotion changed.
