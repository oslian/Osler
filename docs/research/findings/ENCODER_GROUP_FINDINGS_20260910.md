# Fixed-Weight Context Group Localization

Seven32-epoch whole identity checkpoints,12 interventions each in two domains.
All full predictions reproduce cached outputs within1e-5mg/dL. Every mask
preserves glucose itself and all its history; current/mask/age/presence of
the removed channels are cleared consistently. No retraining or selection.
Report: `artifacts/encoder_groups_20260910.json`.

Median paired patient-equal MAE change after removal; negative means removal
helps. Counts are significant improve/harm among seven exploratory95% CIs.

| Removed group | Hospital delta | improve/harm | MIMIC delta | improve/harm |
|---|---:|---:|---:|---:|
| Cardiovascular/perfusion | -0.114 | 2/0 | +0.191 | 1/3 |
| Renal/urinary | +0.097 | 0/1 | -0.303 | 3/0 |
| Acid-base/electrolyte | +0.473 | 0/4 | -0.228 | 3/0 |
| Non-glucose endocrine/metabolic | +0.058 | 0/2 | -0.046 | 3/0 |
| Hepatic/synthetic | -0.110 | 3/0 | -0.159 | 3/1 |
| Gastrointestinal/pancreatic | -0.022 | 1/0 | approximately0 | 0/1 |
| Hematologic/coagulation | +0.775 | 0/5 | +0.567 | 0/5 |
| Immune/inflammatory | +0.002 | 0/2 | +0.195 | 0/4 |
| Cardiac injury | -0.126 | 1/0 | 0 | 0/0 |
| Musculoskeletal injury | -0.123 | 0/0 | -0.170 | 4/0 |
| All non-glucose | +0.441 | 0/3 | +0.714 | 2/4 |

Respiratory is a no-op in both roles: this lab-only contract has no observed
respiratory inputs. MIMIC cardiac-injury removal is likewise a no-op. No-op
cannot be described as absence of organ information in raw data or biology.

## What This Localizes

The acid-base/electrolyte block has opposite sensitivity patterns across the
two datasets. Renal and non-glucose endocrine show weaker analogous patterns.
Blood/coagulation removal harms5/7 in both, so learned full-model predictions
do use helpful non-target context in this conditional masking test.
There is no group whose deletion significantly helps7/7 in both domains.
Wholesale non-target removal also fails to improve consistently.

This is compatible with mixed transferable/nontransferable dependencies, not
one globally harmful organ. It also clarifies why separately TRAINED own-only
can be better while removing inputs from a whole-trained model is worse:
those are different weight/input interventions. Never equate them.

## Limits And Next Discriminating Test

Masking is out-of-distribution and changes measurement-support signals as well
as values; it does not separate physiology from measurement practice, identify
causal organs, prove a unique root cause, or justify automatic feature removal.
The checkpoint groups are functional variable groups, not literal organ nodes.
Seven seeds reuse patients, CIs unadjusted; MIMIC remains inspected validation.

A legitimate follow-up must distinguish value dependence from support/age
dependence and train the selected contract on source data, rather than select
MIMIC-beneficial deletions. No release or calibrator changed; no promotion.
