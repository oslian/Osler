# MAP Increment Decomposition

Reused frozen original native-anchor predictions, verified all parent bindings,
and exactly reproduced existing full-own and full-support paired statistics.
No retraining, selection, runtime or calibration changes.

For each patient, decompose absolute-error contrasts in this fixed order:
total = (support - own) + (full - support). Support retains own MAP numbers
and other observation masks/ages; full adds other numbers. Also directly test
(full - support) - (support - own), using common 20,000 patient bootstrap draws.
This avoids comparing significance counts as though they were effect differences.

| Role | Total contrast | Support contrast | Numerical contrast |
|---|---:|---:|---:|
| Fitting patients | -0.29150 | -0.24724 | -0.03148 |
| Tune patients | -0.02382 | -0.02925 | -0.00911 |
| New source patients | -0.03236 | -0.00463 | -0.01519 |
| Held-out hospitals | +0.09549 | +0.09570 | +0.00997 |

Values are seven-split medians of patient-equal error differences, mmHg.
Negative means improvement. Medians do not necessarily add, although every
row and each split's means satisfy the decomposition exactly.

The support increment fits better than the subsequent numerical increment in
all seven paired fitting comparisons; median contrast difference +0.21734.
Support-own fitting gain is 7/7; full-support fitting gain 4/7. Both lose
significant gains on tune, new source and hospitals (0/7 each).

The correction to interpretation is important: most fitting advantage in this
ordered comparison is already present WITHOUT other numerical physiological
values. Therefore fitting full-own success is weak evidence for learnt
cross-organ numerical dynamics, even before generalization is considered.
This is not proof the model only learns workflow. Masks and ages can reflect
physiology, and separately fitted trees co-adapt; the ordered contrasts are not
causal fractions or a factorial isolation of all mechanisms. Own also has fewer
input features despite equal tree parameter caps. Fit confidence intervals are
resubstitution diagnostics, not out-of-sample scientific confirmation.

This narrows the remaining question to whether numerical cross-history offers
incremental transferable benefit, rather than assuming a large physiological
benefit is merely lost at deployment. It does not prove information absence or
justify deleting observation metadata. The full anatomical JEPA architecture
is not tested here. No promotion or new external lockbox.

Evidence: `artifacts/map_native_increment_decomposition_20260912.json`.
