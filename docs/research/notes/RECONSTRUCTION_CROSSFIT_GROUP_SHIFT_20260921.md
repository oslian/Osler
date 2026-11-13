# Cross-Fit Group Shift: Fixed-Prediction Diagnosis

## Question And Scope

Does the selected/broad reconstruction gap reflect different hospital mixtures,
or different gains within the same hospital components? Does a component-equal
sign reversal depend on very small groups?

Use all five frozen outer prediction files from the completed fresh-encoder
experiment. Verify parent dependencies and prediction hashes, unique
patient-target pairs, and reproduce every parent patient/component mean to
1e-12. No fitting, new selection, removal of adverse groups, or gate changes.
The original empty-history abstention remains outside error calculations and
inside the parent's coverage accounting.

This remains masked CURRENT reconstruction, not future forecasting. Process
control preserves measurement-process inputs but removes physiological values;
population control is the fitting-population center, not persistence.

## Decomposition

For common components, let selected/broad patient fractions be wa/wb and their
paired error differences be ma/mb. Negative error difference favors real
physiology. The exact symmetric decomposition is:

```
selected mean - broad mean
  = sum((wa + wb) / 2 * (ma - mb))
  + sum((wa - wb) * (ma + mb) / 2)
```

The first term is within-component difference; the second is composition.
This is descriptive arithmetic, NOT causal attribution or matched-patient
physiology. It has no sampling-confidence claim. Common-support fractions
are reported separately, not silently assumed to be 100%.

Across 33 targets against process control, 17 have a positive common-support
selected/broad mean gap, 19 a positive within-component term, and 15 a positive
composition term. These counts overlap and are not significance tests.

| Target | Selected minus broad mean gain | Within component | Composition |
|---|---:|---:|---:|
| Albumin | +0.13668 | +0.09664 | +0.04003 |
| BUN | +0.02729 | +0.02630 | +0.00100 |
| Hematocrit | -0.02106 | -0.01535 | -0.00571 |
| Total protein | +0.00137 | +0.06018 | -0.05882 |
| Glucose | +0.02307 | +0.02533 | -0.00226 |
| Creatinine | +0.05824 | +0.06369 | -0.00545 |

All six examples have full common-component support. Units are the parent
normalized target units, not clinical units or percentages. A positive term
means less benefit in selected patients, not necessarily absolute harm.

## Small-Sample Sensitivity Versus Larger-Group Regression

Selected albumin, BUN, and hematocrit each improve against process control in
7/8 components. Their component-equal mean stays negative after leaving out
any single component. This does NOT override the failed prespecified statistical
screen, but it distinguishes insufficient evidence from widespread mean harm.

Selected total protein has a patient-mean gain of -0.16948 but component-equal
difference +0.11734. All four positive component means have fewer than 30
patients. The largest positive mean (+1.09947) has only one patient. The
leave-one-component-out mean ranges from -0.02297 to +0.17287. Thus the aggregate
sign is sensitive to thin support; this is not sufficient evidence of systematic
harm in a well-supported hospital population. Do not delete that patient or
component to make the screen pass.

Conversely, thin support cannot explain every failure:

- Selected creatinine has a 156-patient component with mean difference
  +0.10288, while the same component's 2463 broad patients average -0.01098.
  Its overall equal-component difference remains positive after dropping any
  one component (range +0.00172 to +0.03358).
- Selected glucose has positive means in components of 167 and 152 patients
  (+0.05701 and +0.01638). Broad glucose also worsens in a 1890-patient
  component (+0.06404). These are numerical regressions, not new significance
  claims or proof of a particular biological mechanism.

The diagnosis therefore cannot be reduced to hospital-mixture imbalance alone,
nor to small selected samples alone. Within-component differences can still
reflect case mix, timing, measurement support, noise, or model limitations.

## Decision And Next Discriminating Work

Keep the original screen and all patients unchanged. Do not lower thresholds,
route using these post-outcome subgroup results, or promote any target.
The next comparison should measure whether actual input support, observation
ages and label distributions differ within these components, using the SAME
fixed outer examples. It must distinguish available-input differences from
residual mapping differences before proposing another encoder/head change.
Any resulting model change needs separate validation; this source study is
already development data, not a new external lockbox.

Three tests verify pure composition, pure within-component differences, and
retention of a small adverse component. Report:
`artifacts/native_reconstruction_crossfit_group_shift_20260921.json`.
No runtime, weights, registry, uncertainty artifact or promotion changed.
