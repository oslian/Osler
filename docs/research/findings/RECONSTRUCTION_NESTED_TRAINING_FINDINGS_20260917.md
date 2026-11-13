# Independent Selection: Outcome and Failure Localization

## Scope

Masked current-value reconstruction, not future forecasting or runtime promotion.
All 39 target statuses remain visible. Training retained 33 targets; 28 have at
least 30 patients in fit, encoder-selection, head-selection and evaluation.
The other 11 are unsupported for this four-role experiment, not declared failures.

## Completed comparison

Three arms completed eight epochs. Separating encoder selection from head
selection selected the same 8/4/1 checkpoints as the previous experiment.
All selected model tensors were bitwise identical, permitting verified feature
cache reuse. No inference cache was reused under different weights.

For 28 supported targets, original/median/five ridge heads were selected using
only head-selection patients. Choices were saved before evaluation. Mean
target-normalized MAE was 1.797586 original versus 1.785098 selected. Across all
33 trained targets (five unsupported retained original diagnostically), it was
2.934232 versus 2.923637. These are development summaries, not clinical units.

Nominal paired evaluation gains: AST, bicarbonate, phosphate.
Nominal harms: BUN, respiratory rate, WBC. No complete promotion follows.

## Selection-margin diagnostic

The fixed selected heads were refit on the same fitting data and all selection
scores reproduced within 1e-10. One row per patient-target was verified before
5,000 patient bootstrap draws. No new head or acceptance threshold was selected.

Of 22 changed heads, 20 have descriptive selection difference intervals crossing
zero. All three nominal evaluation harms fall in this group:

| Target | Selection MAE delta | Selection 95% interval | Evaluation MAE delta |
|---|---:|---|---:|
| BUN | -0.0440 | [-0.1372, 0.0454] | +0.1035 |
| Respiratory rate | -0.0792 | [-0.1841, 0.0217] | +0.0652 |
| WBC | -0.0493 | [-0.1711, 0.0667] | +0.1890 |
| Bicarbonate | -0.0952 | [-0.1765, -0.0169] | -0.0693 |
| Phosphate | -0.1221 | [-0.1879, -0.0563] | -0.0552 |

Negative means lower error than the original reconstruction head.
These selection intervals are post-selection and optimistic: the same patients
chose the winner. They cannot be treated as an independent statistical gate.
Evaluation has already been examined and intervals are not multiplicity-adjusted.

## Interpretation and next repair boundary

Separating data roles does not repair the minimum-MAE decision rule by itself.
That experimental rule replaces heads without establishing that their apparent
advantage is reproducible. This is a concrete selection weakness, not proof that
all body representations are useless or that all errors originate in selection.
It is also not a defect proven in the serving registry, which was not changed.

A defensible next repair must nominate heads on training-side data and confirm
them on disjoint source patients/sites before evaluation, with a prespecified
multiple-comparison policy. Do not retroactively accept only bicarbonate and
phosphate from this evaluation or tune a threshold to eliminate these three harms.
Reduced support after another split is a real cost that must remain explicit.
Physiology-versus-process/placebo evidence, external replication and uncertainty
remain necessary for promotion. This diagnostic does not supply those gates.

## Evidence

- `artifacts/native_reconstruction_nested_training_20260917.json`
- `artifacts/native_reconstruction_nested_heads_20260917.json`
- `artifacts/native_reconstruction_nested_margins_20260917.json`
- Six partition and selection regression tests passed.

Release weights, runtime, intervals and promotion status are unchanged.
