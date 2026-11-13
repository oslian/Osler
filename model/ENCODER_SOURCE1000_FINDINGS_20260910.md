# Source Expansion and Failure Location

## Scope

Glucose@6h only: availability-safe observed anchors and actual 5-7h future
measurements. This is not evidence for all release targets or causal physiology.
Frozen protocol: ENCODER_LEARNING_CURVE_PROTOCOL_20260910.md.

## Completed Experiment

Source selection excluded original development patients, original held-out
patients, and every patient with any stay in held-out hospitals. Before looking
at laboratory labels, a deterministic hash selected 4,000 new patients from
91,957 eligible source patients across 131 hospitals. Of these, 2,711 had
58,668 valid labels. Patient and hospital role overlaps are zero.

The first learning-curve stage adds 750 label-eligible patients to each original
250-patient fit. All seven seeds have 1,000 fitting and the same 136 tuning
patients. Four matched own/whole x bounded/identity arms completed 32 epochs,
16 steps per epoch, batch32. The reference is the 250-patient observed-step-mask
run, not the old unmasked history experiment. Hospital and MIMIC query hashes
are unchanged. Checkpoint selection uses source tuning patients only.

Median patient-equal MAE across seven seeds (mg/dL; identity decoder):

| Input | eICU hospital, 250 | eICU hospital, 1000 | MIMIC, 250 | MIMIC, 1000 |
|---|---:|---:|---:|---:|
| Own glucose history | 36.679 | 36.617 | 31.690 | 31.211 |
| Whole laboratory history | 38.070 | 37.116 | 33.497 | 32.286 |

Paired patient-bootstrap significant improvement versus the matched250 arm:
whole7/7 hospital and6/7 MIMIC; own2/7 hospital and5/7 MIMIC. No significant
harm in those comparisons. However whole-vs-own still passes0/7 in both roles.
Whole source-tune MAE remains numerically worse than own in all seven seeds.

## Diagnosis

The earlier fit-gap audit already localized failure BEFORE external transfer:
whole beats own on fitting patients in6/7 seeds but loses on source-tune
patients in7/7 (significantly6/7). Thus external harmonization cannot be the
sole explanation. Expanding source patients now improves the whole model,
supporting sensitivity to source support/composition. It does not isolate
sample count from population composition, or prove either is the sole cause.

Confirmed separate contributors are the direct decoder's amplitude cap and
unsupported ionized-calcium input contract; controlled repairs improved errors.
The empty-history mask defect was fixed structurally but did not yield stable
accuracy gains. Correcting a software invariant is not itself forecast proof.

Current best-supported residual failure: additional non-target context gains
on training patients do not become stable incremental prediction on unseen
patients, even inside the source database. This is an observed generalization
gap, not proof that cross-organ information is absent or that JEPA is incapable.

## Limits and Decision

- Fixed compute means fewer updates per patient after expansion. No convergence
  or data-size ceiling claim.
- Additional patients are broader than the inherited bounded cohort. Count and
  composition are not perfectly isolated; source support may also expand.
- Warm-start pretraining patient overlap has not been excluded.
- Seven seeds share data; they are not seven independent external cohorts.
- MIMIC is repeatedly inspected validation, not an untouched lockbox.
- The prespecified 2,250-patient stage is NOT completed.
- No new conformal calibration, full promotion, or other-target retention test.

Keep candidate_only. Release weights, registry, intervals and serving unchanged.
302 unit tests and git diff whitespace check pass.

## Evidence

- artifacts/encoder_source_expansion_20260910.json
- artifacts/encoder_source1000_20260910.json
- artifacts/encoder_source1000_comparison_20260910.json
- artifacts/encoder_source1000_strata_20260910.json
- ENCODER_FIT_GAP_FINDINGS_20260910.md

Clinical rows and candidate weights remain private local artifacts, not release.
