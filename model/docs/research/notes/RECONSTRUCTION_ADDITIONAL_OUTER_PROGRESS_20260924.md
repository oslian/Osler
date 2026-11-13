# Additional-Domain Outer Inference Progress

## Latest: All Inference and Scoring Complete

Fold3:380292pairs14754people; fold4:318474pairs14577people. All five folds
independently checked together for exact TEST identity/labels, hashes, finite
predictions/scales and no cross-fold patient/component overlap. Total1800214
pairs73670people. Fixed scoring now complete; see
`docs/research/findings/RECONSTRUCTION_ADDITIONAL_FINDINGS_20260924.md`. Earlier pending entries below
are historical. No runtime promotion.

All five-fold/two-arm training completed before first inference. The new fixed
evaluation protocol uses47targets rather than the old helper's hardcoded39.
This family-size correction was specified BEFORE viewing outer results; no
gate was relaxed. Raw-unit per-target errors and fit-scale-normalized macro
errors will be reported separately. No future forecasting claim.

## Fold 0 Complete

380,978patient-target rows,14,759people. Both frozen epoch8models and the
fit-patient center predict identical eligible rows. No fitting, threshold
selection or calibration used these outcomes. Output is private Parquet.

Independent verification: exact test patient-target identity matrix equals
the source-center finite-label matrix restricted to TEST role; no duplicates.
All labels match within float32 tolerance. Predictions and fit scales finite;
prediction SHA and all bound dependencies match. No MAE/winner interpretation
was performed at this intermediate stage. Other four folds remain pending;
all-fold inference will precede the fixed comparison and bootstrap summaries.

No runtime weights, registry, intervals or promotion changed.
Private metadata: `data/research_recovery/additional_development_domains_20260921/outer_20260924/fold0.json`.

## Fold 1 Complete

356,944 patient-target rows, 14,735 people. The original inference process
completed normally without interruption or changes to frozen settings.
Independent verification checks the exact TEST-role patient-target matrix,
absence of duplicates, raw labels against the source centers, positive fit
scales, finite predictions, prediction SHA and every bound dependency hash.
All checks passed. No MAE or winner interpretation performed.

Two of five outer folds are complete: 737,922 rows across 29,494 people.
Folds 2, 3 and 4 remain pending. These are inference counts, not accuracy or
promotion counts. Runtime weights, registry and intervals remain unchanged.
Private metadata: `data/research_recovery/additional_development_domains_20260921/outer_20260924/fold1.json`.

## Fold 2 Complete

363,526 patient-target rows, 14,845 people. Normal process completion and
independent exact TEST patient-target identity verification passed, including
duplicate exclusion, source-label matching, positive scales, finite predictions
and all bound file hashes. Training weights are the previously certified fold0
equivalent fit; this is a distinct outer patient group, not another independent
training replication. No model choice or score interpretation used these rows.

Three of five outer folds are complete: 1,101,448 rows across 44,339 people.
Folds 3 and 4 remain pending before the fixed all-target scoring. No promotion,
runtime, calibration or interval changes.
Private metadata: `data/research_recovery/additional_development_domains_20260921/outer_20260924/fold2.json`.
