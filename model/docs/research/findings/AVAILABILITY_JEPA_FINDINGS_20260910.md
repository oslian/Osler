# Availability-Safe JEPA Findings

## Decision

Completed re-encoding and seven source-only readout fits on the SAME availability
anchors and glucose labels as the previous ridge diagnostic. No architecture
replacement or runtime promotion. Full checkpoint input width is preserved;
only verified laboratory channels are populated, not all possible bedside data.

Final evidence is `artifacts/availability_jepa_20260910_v2.json`.
The first artifact `artifacts/availability_jepa_20260910.json` is retained:
it used an incomplete equivalent-unit whitelist. V2 accepts K/mcL, mcg/dL,
mcU/ml and mOsm/kg H2O without numeric conversion, using the raw unit inventory,
not outcome-selected model parameters. Neither is an untouched validation.

## Input Contract

Native lab event IDs, draw time and result-entry time retained. Snapshot state
uses the latest measured known result. A delayed older draw cannot overwrite a
newer known draw. Exact 48/24/12/8/6/4/3/2/1/0h snapshots use each lag's own
availability cutoff; absent early history is missing, not a repeated first row.
Current glucose reconstruction matches the prior rebuilt anchors within0.001
mg/dL. No old 311-feature vectors or old labels are carried over.

42 eICU laboratory variables (131,556 events) and40 MIMIC variables (312,544
events) enter the53-variable schema. Not every variable is present at each query.
Unverified bedside vitals, urine and treatments stay missing. Ionized calcium
mg/dL is not yet converted in eICU; unitless MIMIC INR is rejected. Other absent
channels include uncatalogued aliases and insufficient support. Therefore
input support differs between databases; this is not complete clinical coverage.
No treatment was added or declared absent: the frozen input schema has no new
treatment channels in this experiment.

Same data: development7,140 pairs/386 patients; eICU hospital6,077/364;
MIMIC6,616/889. Source seeds change fit/tune patients only, not external cohorts.
Frozen checkpoint source provenance is inherited; this is not a new independent
evaluation of encoder training. Source-only refers to the newly fitted readouts.

## Frozen JEPA Results

Patient-equal glucose MAE, mg/dL:

| Input to existing checkpoint and existing shared head | eICU hospital | MIMIC |
|---|---:|---:|
| Persistence | 45.878 | 37.449 |
| Glucose-only, true masks/ages | 42.463 | 34.689 |
| All accepted labs, true masks/ages | 41.611 | 34.926 |
| All accepted labs, old all-one/zero-age encoding | 39.158 | 33.126 |

Truthful full-lab frozen model beats persistence in both domains:
eICU delta-4.267 (95%CI -7.099,-1.819); MIMIC-2.523 (-4.064,-0.775).
Full-lab vs own: eICU-0.853 (-1.162,-0.540), MIMIC+0.237 (-0.083,+0.567).
Thus there is no stable external incremental whole-body benefit.

Changing masks/age with frozen weights worsens both domains relative to the
old encoding of the SAME legal numerical inputs. This is consistent with an
encoding-distribution mismatch, not proof that concealing missingness is correct.
Do not deploy the old mask convention as a scientific fix.

## Source-Refitted Readouts

Same311 feature widths, source-fitted standardization, patient-equal loss
weights and source-only alpha1/10/100 selection. Seven-seed median MAE:

| Ridge readout | eICU hospital | MIMIC |
|---|---:|---:|
| Own glucose JEPA features | 37.181 | 33.467 |
| Full laboratory JEPA features, true masks/age | 43.891 | 56.979 |
| Full laboratory JEPA features, old mask encoding | 43.461 | 57.480 |

Full truthful readout beats own0/7, frozen full head0/7 in both datasets.
It beats persistence1/7 eICU,0/7 MIMIC. Refitting the large feature bundle
does not repair external transfer and can cause major extrapolation harm.
The311 features contain raw variables as well as latent features; these results
must not be attributed solely to shared latent quality.

## Failure Localization

`artifacts/availability_jepa_shift_20260910.json` holds non-interventional
feature-shift diagnostics with no model selection. Under seed7's source scaler,
MIMIC full features have0.860% values beyond8 source standard deviations versus
0.0046% own-only. The largest p95 columns include pH20.47, LDH19.17,
bilirubin17.01 and ionized calcium15.91. Source absence of ionized calcium is
itself a support mismatch; a standardized value is not necessarily a clinically
extreme laboratory value. These are associations with failure, not a proven
single cause. Current readout uses uncapped StandardScaler and Ridge.

The established software repair is availability-safe laboratory reconstruction.
Remaining hypotheses: frozen encoder/input semantics mismatch, unsupported
cross-database feature variation and high-dimensional readout overfit. This
experiment does not prove that more data/architecture can never help, nor that
JEPA/self-learning has been repaired.

## Boundaries and Verification

- No end-to-end encoder retraining, treatment replay or self-learning tested.
- No new conformal, care-unit/time, shuffled-feature placebo or full promotion.
- MIMIC is already inspected validation; a new final lockbox is still required.
- All source/patient rows and feature caches stay in temporary local storage.
- 289 regression tests pass; no release weights, intervals, registry or serving
  behavior changed. Structural correctness is not promotion.

Next discriminate supported-channel mismatch from encoder mismatch on source
development data before another wide readout sweep. Full availability-safe
bedside/treatment observation remains unfinished and must not be described as
completed merely because the53-slot tensor can now be constructed.
