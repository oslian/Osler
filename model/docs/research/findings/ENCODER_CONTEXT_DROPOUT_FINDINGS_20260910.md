# Context Dropout Results

All28 source-only32-epoch fits completed under the fixed protocol. Seven
seeds, own/whole and bounded/identity arms, availability-safe glucose@6h.
Every own-arm training log and prediction exactly reproduces the original
budget experiment. Dropout is training-only, no artificial inference values.

Median patient-equal MAE, mg/dL:

| Arm | Hospital original | Dropout | MIMIC original | Dropout |
|---|---:|---:|---:|---:|
| Own bounded |41.322|41.322|34.415|34.415|
| Own identity |36.559|36.559|31.238|31.238|
| Whole bounded |42.388|42.085|35.572|35.536|
| Whole identity |38.202|37.479|33.368|33.128|

Paired patient bootstrap comparisons against the same original arm:

- Whole bounded: hospital improvement4/7, harm0/7; MIMIC improvement3/7,
  harm1/7.
- Whole identity: hospital improvement3/7, harm0/7; MIMIC improvement3/7,
  harm1/7.
- Whole versus own:0/7 significant improvement in either activation or domain.
- Whole identity versus persistence:7/7 pooled in both domains, but not all
  care-unit/time strata. MIMIC TSICU0/7, CCU2/7, MICU6/7, MICU/SICU5/7,
  CVICU7/7, SICU7/7. The2014-2016 subgroup remains0/7.

The fixed augmentation yields modest changes, not a stable resolution of
negative transfer. It does not establish that absence of regularization was
the sole root cause. It also does not establish that every context masking
scheme is ineffective. No external hyperparameter selection was performed.

Existing sources are repeatedly inspected validation sets. Period subgroup
audits are not a forward-time holdout. No calibrator, runtime weight, source
selector or promotion changed.299 unit tests pass.

Evidence: artifacts/encoder_context_dropout_20260910.json,
artifacts/encoder_context_dropout_comparison_20260910.json,
artifacts/encoder_context_dropout_strata_20260910.json.
Protected candidate checkpoints remain outside version control.
