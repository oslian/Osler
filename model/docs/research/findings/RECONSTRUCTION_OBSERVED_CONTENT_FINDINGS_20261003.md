# Observed Content: Event Encoding Versus Final Body

## Question And Result

For the SAME already observed numeric value, is it easier to decode its
last-observed event representation or the final persistent body state?
This is not masked-value inference, future forecasting, or a promotion trial.

The fixed experiment completed all five source folds. All53 schema variables
were considered:36 had support in at least one fold,35 in all five. Anion gap
had support in folds0/1 only. The other17 were unsupported, not model failures.
One hash-selected masked context per patient/fold; repeated fold losses were
averaged per patient before scoring. There were177 fitted variable-folds,
1062 readers,88122 observed-value queries and2048 unique selection patients.
The five folds are not five independent patient cohorts.

| Reader | Event MAE, patient | Final body MAE, patient | Event MAE, component | Body MAE, component |
|---|---:|---:|---:|---:|
| Ridge | 0.911219 | 1.259259 | 0.962995 | 1.291298 |
| HGB | 0.628675 | 0.893199 | 0.680428 | 0.967497 |

These are target-macro errors in fold-normalized units, not clinical units or
accuracy percentages. Both representations have64 latent coordinates plus
the SAME159 process features. Event encoding has less subsequent context.
Process-only errors were1.533078 Ridge and1.059718 HGB (patient macro).

Event had lower patient-mean error for all36 supported variables with both
readers. The prespecified patient/component paired screens, including
53*6 multiplicity correction, passed event-versus-body for23 Ridge and19 HGB
variables. The intersection is16:

chloride, glucose, heart_rate, hematocrit, hemoglobin, inr, lactate, map,
o2sat, platelets, potassium, ptt, respiratory_rate, sodium, temperature, wbc.

All16 have five-fold support. Fourteen also beat process-only with both
readers under the same screens; glucose and ptt do not meet that stronger
intersection. Do not count these screens as deployed forecast cells.

## Selected Examples, Full Family Retained In JSON

| Observed variable | Ridge event/body | HGB event/body |
|---|---:|---:|
| BUN | 0.331416 / 0.432080 | 0.355173 / 0.420840 |
| Creatinine | 0.513582 / 0.793723 | 0.487809 / 0.630285 |
| Bicarbonate | 0.317121 / 0.501138 | 0.393049 / 0.526704 |
| Glucose | 0.458629 / 0.699612 | 0.460243 / 0.688468 |
| INR | 0.545032 / 1.088726 | 0.301552 / 0.803635 |
| Heart rate | 0.196710 / 0.405269 | 0.174499 / 0.391713 |

Mean improvement is not a full screen. BUN, creatinine and bicarbonate are
not in the two-reader event/body screen intersection.

Unsupported throughout this bounded sample: amylase, bnp, ck_mb, cortisol,
crp, crp_hs, esr, ferritin, free_t4, ionized_calcium, ldh, myoglobin, ph,
prealbumin, serum_osmolality, uric_acid, urine_output. Counts and support
reasons are retained per fold; missing values were never scored as zero.

## What Changed In The Diagnosis

E17 could not show a universal advantage from averaging events for inferring
an UNKNOWN masked target. This experiment asks a different question: retaining
KNOWN inputs. Its evidence supports reduced accessible observed-content
fidelity between event encoding and final persistent state. The difference
persists across a linear and a nonlinear reader, not just one head.

It does NOT establish irreversible information loss. Coordinates and finite
reader capacity remain confounds. The body was optimized for masked inference,
not exact storage of every known input; retaining a number and reasoning about
a missing number are different capabilities. Direct raw copying would be exact
here, but would not solve the actual unknown-target task. E18 and E15 remain
important counterexamples against declaring a simple skip/alignment repair.

The first failure cannot yet be localized more finely than the path between
the last-observed event representation and final body: the immediate update,
later dynamics, and assimilation of other observations are not separated.
The next diagnostic must compare these stages for the same known content,
with separately fitted readers. Merely applying the final readout at earlier
steps or repeating mean pooling would repeat already ambiguous experiments.

No new encoder repair has been trained or validated by this result. No
release weights, registry, intervals or serving behavior changed. No causal,
clinical, information-theoretic impossibility or untouched-lockbox claim.

## Integrity And Reproducibility

- Session93068 completed normally; no partial-run model selection.
- Verification reproduced all177 saved groups/1062 reader outputs, checked
  role disjointness, patient/context/truth identity, support and hashes.
- All531 saved scalers have finite parameters and positive scales.228 contain
  tiny negative variances, minimum-3.6261e-38, within constant-roundoff bounds
  and guarded with scale1. These warnings are not nonfinite prediction errors.
-19 focused tests passed. They validate software invariants, not physiology.
- Source-development data is reused. Small source-component counts and the
  parametric paired-test assumptions limit inferential strength.
- Bindings describe the actual working-tree code. Some dependencies remain
  uncommitted historical work; this is not a clean-checkout release package.
- Patient features, IDs, models and predictions remain in ignored private
  data. Only aggregate evidence is published in repository artifacts.

Evidence: `RECONSTRUCTION_OBSERVED_CONTENT_PROTOCOL_20261003.md`,
`artifacts/native_reconstruction_observed_content_fold{0..4}_20261003.json`,
`artifacts/native_reconstruction_observed_content_assessment_20261003.json`,
`artifacts/native_reconstruction_observed_content_verification_20261003.json`.
Code: `scripts/probe_observed_content.py`, `scripts/verify_observed_content.py`.
