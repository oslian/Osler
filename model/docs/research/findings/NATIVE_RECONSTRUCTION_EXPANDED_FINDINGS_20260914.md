# Expanded Shared-Body Reconstruction Results

## Completed Experiment

Training: `artifacts/native_reconstruction_expanded_training_epoch8_20260913.json`.
Evaluation: `artifacts/native_reconstruction_expanded_evaluation_20260913.json`.
All evaluation dependency hashes were verified after completion.

Both shared-body arms completed eight epochs, 35,997 updates each epoch
(287,976 per arm), covering all 25,511 source-eligible fit patient-target pairs
each epoch. Thirty-nine targets were chosen using source fit/tune support only.
The same fully target/sibling-masked persistent state architecture was used.
Patient roles: 923 fit, 335 tune, 311 evaluation, 194 hospital-heldout.

Real-label source selection chose epoch7, normalized target-equal tune MAE
1.921813 versus initial2.085252 (about7.84% lower). The placebo arm chose
epoch0; all its trained epochs were worse on real tune labels. The continuation
retained optimizer and random states; no mid-run setting changes were made.
Heldout errors were evaluated only after both arms finished.

## Nominal Paired Evidence

Fourteen targets have supported, unadjusted95% paired patient-bootstrap error
intervals below zero against BOTH selected placebo and population in BOTH roles:

- albumin
- alkaline_phos
- alt
- ast
- bicarbonate
- chloride
- glucose
- heart_rate
- hematocrit
- hemoglobin
- platelets
- ptt
- total_protein
- wbc

Selected placebo is epoch0 and therefore essentially the population median
(floating-point rounding aside). Those two comparisons are not independent
confirmations; the trained null control failed source selection.

| Target | Evaluation population -> real MAE | Hospital population -> real MAE |
|---|---:|---:|
| Albumin | 0.8219 -> 0.5494 | 0.8188 -> 0.5580 |
| Bicarbonate | 1.0627 -> 0.9038 | 0.9920 -> 0.8591 |
| Glucose | 1.4075 -> 1.3315 | 1.7620 -> 1.6921 |
| Hemoglobin | 0.7693 -> 0.6522 | 0.8195 -> 0.6979 |
| Total protein | 0.9053 -> 0.6021 | 0.9658 -> 0.6103 |

All numbers in this table are normalized units, NOT physical-unit errors.
Across all39 targets mean MAE is2.217634->2.076808 in evaluation and
3.159198->3.057649 in hospital. Heavy-tailed sparse markers influence these
averages; per-target errors and paired intervals are the primary evidence.

## Remaining Failures And Limits

- Creatinine and MAP do not show significant population improvement in both
  roles. BUN, calcium and phosphate improve in evaluation but not hospitals.
- BNP is nominally worse in evaluation; magnesium is nominally worse in
  hospitals. Shared training is not uniformly beneficial.
- Fibrinogen, LDH and TSH have only26/27/17 hospital patients and remain
  statistically unsupported, not demonstrated equivalent to baseline.
- These are nominal per-target intervals with no familywise/multiple-testing
  correction. One reused eICU split is not seven-seed confirmation or a new
  database lockbox. No target is promoted on this evidence alone.
- This is fresh-current-value masked reconstruction. There are no new future
  forecast cells or calibrated intervals. Masks induce different inferred
  realizations of the same shared-state model; this does not prove every output
  comes from an identical unmasked numerical state.
- No ablation yet separates raw-value physiology from remaining observation
  support/cadence effects. Reconstruction gain is not a causal organ mechanism.

## What This Locates

The previous tiny pilot was an inadequate test of whole-body learnability:
direct training exposure was sparse, and every evaluation role was below the
30-patient reporting threshold. The larger, completely covered run now yields
nominal heldout gains for multiple targets without changing architecture.
That supports the importance of experimental support and optimization exposure,
not a claim that these are the sole causes of every historical failure.
Patient count, sampling and budget changed together relative to earlier work;
their independent causal contributions are not identified.

Next require repeat-split evidence and source-defined controls for observation
process and shared-state dependence; preserve all39 targets rather than selecting
only these14 post-hoc winners. Future-prediction and runtime gates remain separate.
Release weights, registry, intervals and serving behavior are unchanged.
