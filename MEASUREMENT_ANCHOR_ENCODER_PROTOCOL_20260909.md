# Measurement-anchor encoder diagnostic

Frozen before outcome evaluation on 2026-09-09. This is a bounded diagnostic
for `creatinine@6h`; it cannot promote a runtime source.

## Questions

1. Does anchoring at a real creatinine measurement, instead of carrying an older
   creatinine into a regular 3h grid anchor, improve future-value prediction?
2. With identical patients, labels, readout family and capacity, does a frozen
   whole-body JEPA representation outperform a target-only JEPA representation?

The target-only representation replaces every non-creatinine value at every
history position with the checkpoint's normalized median (zero). It retains the
same time and identity metadata as the whole-body arm. It therefore removes
patient-specific non-target measurements without creating a new missingness
marker the deployed checkpoint was not trained to interpret.

## Data and timing

- Source/development: eICU common canonical measurement ledger, 5,000 patients.
- Repeated external validation: MIMIC-IV common ledger, 5,000 encounters. MIMIC
  is not an untouched lockbox.
- Fresh mode: anchors are actual observed creatinine measurement timestamps,
  6-96h after episode start. Duplicate source rows at one timestamp form one
  anchor. Creatinine age must equal zero.
- Carried mode: existing regular 3h-grid anchors, used only as a sensitivity
  reference.
- Both modes require an observed future creatinine 5-7h after anchor, with the
  recorded delta agreeing with its timestamp. Future measurement time and value
  never enter model input.
- Within each database, only patients with eligible rows in BOTH modes are used.
  The eICU hospital fold and patient roles are fixed from fresh-mode support and
  reused unchanged by carried mode. The MIMIC patient set is also identical.
- This controls patient membership and split. It does NOT pair identical anchor
  times or outcomes across modes, so cross-mode MAE differences are descriptive,
  not causal estimates of freshness.

## Fixed predictors

All fitted delta heads are patient-weighted HistGradientBoostingRegressor models
with the project's existing fixed configuration. Every arm includes the same
seven target-own trend features.

1. `own_history`: seven measured creatinine history features.
2. `target_only_encoder`: own history plus 256 named frozen JEPA latent features
   generated after all non-creatinine patient values are neutralized.
3. `whole_body_encoder`: own history plus the same 256 latent dimensions from
   the unmodified legal pre-anchor state.
4. `whole_body_placebo`: own history plus a time-matched different patient's
   whole-body latent.
5. `persistence`: anchor creatinine unchanged.

Primary within-mode comparison is whole-body versus target-only. Whole-body must
also beat wrong-patient latent, own history and persistence. Target-only versus
own history diagnoses whether the frozen encoder adds useful own-target dynamics.
All heads use the same fit rows and update budget. No architecture tuning follows
MIMIC inspection.

## Evaluation and interpretation

- One representative anchor per patient in patient, hospital and MIMIC tests.
- Patient-level paired bootstrap CI (1,000 repetitions) plus seven patient
  resamples. Seven resamples are not seven independent training runs.
- Supported care-unit breakdown requires at least 30 patients.
- Exploratory CIs are unadjusted for multiple comparisons. A favorable diagnostic
  does not authorize promotion without time holdout, calibrated intervals and a
  new untouched database.
- A whole-body win over target-only supports incremental non-target information
  in this frozen representation/readout. Failure only rejects this finite test;
  it does not prove physiological information absent.
- A fresh-mode improvement with carried-mode failure suggests freshness matters,
  but cannot isolate freshness from the different anchor/outcome distribution.
- No causal or treatment-effect claim is allowed. Runtime weights, registry,
  calibrators and serving behavior remain unchanged.

Aggregate reports and fitted-head hashes may be committed. Row-level examples,
patient identifiers and local fitted diagnostic heads remain outside Git.
