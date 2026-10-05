# Source-confirmed head experiment

Keep nested encoder checkpoints and all 39 target statuses. For each of the 28
four-role supported targets, nominate one of the existing five ridge heads or
fit-median using five-fold GroupKFold on fitting hospital components only.
Refit the nominee on all fitting patients. The representation was trained on
these patients: this head-only CV is nomination, not independent encoder evidence.

Confirm on readout_tune, which did not train or select the current encoder.
Require at least 30 patients and three hospital components. Test paired absolute
error against both original and fit-median, using one-sided t tests of patient
differences and equal-weight hospital-component means. Require negative means
and all four p values below .05/39 (fixed Bonferroni family includes unsupported
targets). A constant zero difference has p=1. Site tests have low power and
normal-mean assumptions; passing is not proof of deployment safety.

Save nominations/confirmation decisions before scoring evaluation. No threshold
tuning, no drops, no replacement if confirmation fails. Evaluate all 33 trained
targets; the other six remain explicit unsupported. This entire dataset has
been inspected historically: independent data roles within the algorithm are
not a new untouched external validation. No runtime promotion or interval claims.
