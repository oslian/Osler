# Extended Budget Diagnostic

Completed28 fits,32 epochs each, with identical source patient sampling,
optimizer, JEPA/value loss, support masks, analyte cache and source-tune
selection. All28 first-eight-epoch logs exactly match the eight-epoch controls.
Both reports bind identical raw/query hashes. No external checkpoint selection.

## Matched Outcomes

Patient-equal MAE, median across seven source seeds, glucose6h mg/dL:

| Arm | Hospital8 | Hospital32 | MIMIC8 | MIMIC32 |
|---|---:|---:|---:|---:|
| Own bounded | 42.112 | 41.322 | 34.536 | 34.415 |
| Own identity | 38.091 | 36.559 | 31.812 | 31.238 |
| Whole bounded | 42.400 | 42.388 | 35.423 | 35.572 |
| Whole identity | 38.331 | 38.202 | 33.087 | 33.368 |

Paired32-vs8 CIs significantly improve/harm, out of7:

| Arm | Hospital improve/harm | MIMIC improve/harm |
|---|---:|---:|
| Own bounded | 6/0 | 1/1 |
| Own identity | 7/0 | 5/0 |
| Whole bounded | 0/0 | 2/3 |
| Whole identity | 1/0 | 1/4 |

Both identity arms still beat persistence7/7 pooled in each domain. Whole vs
own remains0/7 in both. More training is useful for own history, but is not a
sufficient repair of full-context transfer. All own identity selected epochs
are29-32, whereas whole identity spans5-31. This experiment does not establish
optimization convergence or rule out every alternative training schedule.

## Where It Still Fails

MIMIC TSICU identity-vs-persistence0/7 for both views. Own passes7/7 the other
five supported care units and all five period groups. Whole only passes7/7
CVICU and SICU; CCU1/7,MICU3/7,MICU/SICU4/7. Whole2014-2016 period0/7.
These are descriptive, repeatedly inspected subgroups, not pure time holdout.

Source-tune improvement does not imply external improvement. Whole learning
is consistent with nontransferable conditional relationships or selection
overfitting; this is not proof of their unique physiological cause. Increasing
training time alone is now tested and does not close the whole-vs-own gap.
Do not describe the failure as simply eight epochs being too short.

## Evidence And Limits

`artifacts/encoder_budget_20260910.json`: training and pooled outcomes.
`artifacts/encoder_budget_comparison_20260910.json`: exact prefix reproduction
and paired short-vs-long comparisons.
`artifacts/encoder_budget_strata_20260910.json`: all supported/unsupported groups.
Weights and ordered row predictions stay local.296 tests pass.

No runtime, calibration or registry changes. No promotion, pristine lockbox,
multiple-comparison-adjusted inference, clinical or causal claims. Future
work must diagnose which source-learned non-target relationships fail rather
than selecting MIMIC winners or endlessly extending the same optimizer.
