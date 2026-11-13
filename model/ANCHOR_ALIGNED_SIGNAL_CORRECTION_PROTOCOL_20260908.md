# Corrected anchor-aligned attribution diagnostic

Fixed before corrected outcome evaluation. Target: creatinine; nominal horizon:
6 hours. No runtime or promotion changes. This is not an AKI-onset matched trial.

## Timing and data roles

Primary: recorded actual future measurement 5-7 hours after anchor, inclusive.
Secondary: original broad 2-10 hours. Timestamp and recorded delta must agree.
Anchor creatinine remains a real measurement at most 6 hours old. Consequently
the separation between the two measurements can exceed the future horizon;
report anchor measurement age separately. Actual future time selects evaluation
pairs but is NEVER supplied to the predictor. This estimates accuracy conditional
on a future measurement being available, not accuracy for unmeasured outcomes.

Reuse immutable 5000-encounter common-ledger extracts. Hospital and patient roles
are chosen once using broad label support, independently of values. Narrowing
does not reselect hospital folds. Patient support changes between windows, so a
cross-window difference is sensitivity evidence, not a causal effect of timing.
Keep full pre-anchor history even when the selected label is ineligible.

eICU fits all heads. Freeze and save fitted heads before MIMIC evaluation.
MIMIC is repeatedly inspected validation data, not an untouched lockbox. The
pretrained JEPA exclusion of these patients has not been independently verified;
heldout claims apply to new readout fitting, not necessarily pretraining.

## Fixed arms

All learned arms use identical patient-weighted HGB absolute-error delta heads
(60 iterations, 15 leaves, min leaf 40, L2=1, learning rate=.05, seed=20260908).
The baseline is target-own history; persistence is separately evaluated.

1. Own + non-target current values/masks/ages versus own and a same-capacity
   current block replaced by another patient's block.
2. Own + current + lagged history versus own+current, a temporal-only placebo
   preserving own current cross-state, and a full cross-block placebo.
3. Own + pure learned JEPA blocks versus own and a swapped-latent placebo.
   Select named context/belief/target-future/organ blocks from feature contract.
4. Own + full 311-feature release bundle versus own and bundle placebo. This
   arm contains raw current values and base forecast; it is NOT latent-only.

Placebos retain own history and replace the explicitly named added block using
the existing wrong-patient/time-matched donor policy. Same rows and donors for
each within-window comparison. Temporal summaries use the sampled history grid,
not every raw event. No new dependency or neural architecture is introduced.

## Reporting and limits

One representative anchor per patient at evaluation; source patient/hospital and
MIMIC care-unit breakdowns; paired patient bootstrap (1000) and seven resampled
subsets. These are NOT seven independent training seeds. Report all comparisons,
including failure against persistence, with no post-result model selection.
Unsupported care units (<30 patients) are not evidence of passing. Minimum total
support is 100 patients and 100 rows per role; otherwise stop as unsupported.

These are exploratory unadjusted CIs across several arms; any apparent positive
requires confirmatory multiplicity-aware testing, time holdout, and interval
calibration. No pass here promotes a source. Failure of a finite fitted readout
does not prove information absent, encoder information loss, or physiological
unpredictability. Common extraction rules do not eliminate population or clinical
phase differences. Preserve the previous report; supersede its overstrong prose.
