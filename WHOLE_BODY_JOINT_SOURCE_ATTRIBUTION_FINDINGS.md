# Whole-Body Joint Source Attribution

## Question

The high-observation joint JEPA contains partial cross-system signal, but
single-source cardio-renal residual adapters were unstable. This audit asks a
narrower factual question: after a joint model is trained, which *causally
available numeric evidence groups* make held-out creatinine forecasts worse
when removed?

This is not a causal intervention analysis. A source means the module that
provided the freshest as-of value in the aligned state, not an organ causing a
future outcome.

## Method

- 15 body-system modules, 76,682 high-observation as-of rows, 1,124 patients,
  38 hospitals.
- The as-of aligner records `_source_module_<variable>` for every value. This
  prevents duplicate laboratory columns in several module tables from being
  mistaken for independent organ evidence.
- The target's current creatinine is retained. For each source group, only its
  non-creatinine numeric values are masked from both current and history
  inputs; module-presence metadata stays unchanged.
- The fixed full model is compared with the source-ablated view over seven
  patient-held-out splits and one hospital-held-out split.
- A stable contributor requires every patient split to improve with the full
  view, every paired patient bootstrap to pass, and the hospital-held-out
  comparison to pass.

## Result

`metabolic_renal -> creatinine@12h` is the only stable group.

| Scope | Full minus ablated MAE | Result |
| --- | ---: | --- |
| Seven patient-held-out splits, median | -0.0159 | Full input wins in 7/7; paired bootstrap passes 7/7 |
| Hospital-held-out | -0.0709 | 95% paired-bootstrap CI [-0.1262, -0.0380] |

The group contains `aki`, `electrolyte_acid_base`, `endocrine_stress`,
`gi_pancreatic_nutrition`, and `musculoskeletal_rhabdo`.

The individual-source audit found positive but non-generalizing 12-hour
signals for AKI, sepsis, and electrolyte-acid-base data. Grouping the related
metabolic/renal context is what made the held-out hospital effect stable.

## Rejections

- `inflammatory_heme -> creatinine@12h` helped in patient splits but reversed
  in the held-out hospitals.
- `cardiorespiratory -> creatinine@12h` was directionally positive but did not
  pass the paired patient bootstrap.
- `neuro_hepatic_toxic -> creatinine@12h` lacked complete split support.
- One- and six-hour source windows have insufficient cross-hospital exposure
  for a stable claim; this is reported as support-limited rather than treated
  as absent physiology.

## Consequence

This supplies one data-driven candidate for a sparse, target-specific joint
adapter: `metabolic_renal -> creatinine@12h`. It remains **candidate-only**
until a new adapter trained from this group improves the full factual runtime
gate. Nothing in this audit changes the current runtime registry, makes a
causal claim, or authorizes clinical use.

## Adapter Runtime Gate

The attribution result was then tested as a deliberately narrow sparse
adapter. The candidate receives the joint latent plus only the
`metabolic_renal` group and is allowed to alter `creatinine` at 12 hours. Its
input hides current creatinine while retaining BUN and the other valid renal
context, matching the source-ablation contract. Future labels use raw,
timestamped events only, actual per-target elapsed time, quality weights, and
at least two valid target observations.

Both the adapter and a matched no-adapter baseline used the same 76,682 rows,
1,124 patients, 38 hospitals, seven patient splits, and hospital-held-out
gate. Neither qualified for runtime promotion.

| Check for `creatinine@12h` | Baseline | Metabolic-renal adapter |
| --- | ---: | ---: |
| Patient splits where value beats persistence | 1 / 7 | 2 / 7 |
| Patient paired-bootstrap passes | 0 / 7 | 0 / 7 |
| Median patient-level delta vs persistence | +0.0559 | +0.0736 |
| Cross-system ablation passes | 7 / 7 | 7 / 7 |
| Hospital conformal coverage | 0.8707, pass | 0.8707, pass |
| Hospital module-balanced gate | fail | fail |

The group is therefore a **real predictive contributor**: removing it hurts
the joint model across patient and hospital splits. But forcing that
contributor through a sparse adapter does not yet produce a reliable
improvement over persistence. The adapter improved the count of patient
value wins slightly, while worsening the median patient-level error and
failing the paired-bootstrap and module-balanced gates. It is rejected as a
runtime source and stays research-only.

This separates two facts that should not be conflated:

1. Metabolic-renal context contains stable information about a future
   creatinine trajectory.
2. The current joint-JEPA decoder cannot yet convert that information into a
   generalizable, calibrated 12-hour creatinine forecast.

The registry remains useful for future attribution-first experiments, but no
additional group may be promoted without repeating both the provenance audit
and the full factual runtime gate.

Artifacts:

- `whole_body_joint_creatinine_source_attribution_5plus_10k.json`
- `whole_body_joint_creatinine_group_attribution_5plus_10k.json`
- `whole_body_joint_gate_metabolic_renal_baseline_5plus_10k.json`
- `whole_body_joint_gate_metabolic_renal_adapter_5plus_10k.json`

## Observed Treatment-History Context Gate

The joint cohort originally discarded the raw transition tables' `hist_*`
fields before whole-body alignment. This was a factual-observation defect:
the anchor state could contain laboratory values while omitting already
recorded fluid, diuretic, renal-replacement, vasopressor, and nephrotoxin
history. The loader now carries only `hist_*` treatment evidence forward;
`act_*` fields from the forecast window remain excluded.

On the same 76,682 rows, 1,124 patients, and 38 hospitals, giving the shared
body model all 38 available treatment-history fields improved the held-out
hospital creatinine MAE from `0.4469` to `0.4321`, closer to persistence
(`0.4288`). It also made the AKI module's delta essentially neutral
(`-0.00005`). It did **not** pass the factual gate: patient value wins fell
from 2/7 to 1/7, paired patient bootstraps remained 0/7, and the hospital
bootstrap CI still crossed zero (`[-0.0237, +0.0249]`).

This rejects the claim that a broad, generic treatment feature block solves
the joint forecast problem. It supports a narrower next experiment: keep
treatment context out of the shared body latent, then offer only renal-
relevant pre-anchor history fields to the attribution-backed
`metabolic_renal -> creatinine@12h` adapter. That experiment remains
candidate-only and factual; observed treatment context is not a treatment
effect or a causal claim.

Artifact:

- `whole_body_joint_gate_metabolic_renal_treatment_context_5plus_10k.json`

## Targeted Renal Treatment-Context Gate

The follow-up isolated 12 renal-relevant history fields (fluids, diuretics,
renal replacement, nephrotoxins, vasopressors, and albumin, each with its
evidence count). The shared JEPA latent was explicitly prevented from reading
them; only the `creatinine@12h` metabolic-renal adapter received their latest
anchor-time value, observed flag, and measurement age.

This stricter candidate was rejected. The hospital-held-out creatinine MAE
rose to `0.4809` versus persistence `0.4288`, with a harmful paired bootstrap
CI `[+0.0167, +0.0807]`; both AKI and sepsis module-balanced checks failed.
Its conformal coverage was `0.8571`, below the required range.

Together, the broad and targeted experiments identify the remaining issue:
the current eICU `hist_*` fields are administration evidence/counts, not
dose-, route-, formulation-, and rate-resolved therapy trajectories. They are
useful enough to nudge one hospital split toward persistence, but not precise
enough to generalize across patients and hospitals. No treatment-context
candidate is promoted. The next defensible input upgrade is to reconstruct
administration-grade, pre-anchor treatment histories from raw MIMIC
`inputevents` and `emar`, still evaluated as factual context only.

Artifact:

- `whole_body_joint_gate_metabolic_renal_targeted_treatment_5plus_10k.json`

## MIMIC Administration-Grade Factual Validation

The eICU history fields cannot establish whether treatment context itself is
useful because they contain only coarse evidence/counts. The local credentialed
MIMIC-IV v3.1 tables supplied a separate strict as-of test: 640,164 observed
six-hour transitions from 63,307 patients, with pre-anchor `inputevents`,
eMAR, eMAR-detail, and procedure histories. Future `act_*` columns were
excluded from every arm. Open infusions contribute only their current rate;
their final total amount is excluded until the infusion is complete.

Under seven patient-held-out splits plus care-unit and time-held-out gates,
the dose/rate history candidate was validated for `glucose`, `map`, and
`bicarbonate`; it failed the care-unit gate for `potassium` and remains a
fallback there. Representative strict held-out improvements versus the same
no-treatment factual baseline were:

| Target | Care-unit delta MAE, 95% CI | Time-held-out delta MAE, 95% CI |
| --- | --- | --- |
| Glucose | -1.3699, [-2.0605, -0.7257] | -0.8059, [-0.9354, -0.6913] |
| MAP | -0.1615, [-0.3227, -0.0150] | -0.0667, [-0.1064, -0.0327] |
| Bicarbonate | -0.0392, [-0.0585, -0.0180] | -0.0108, [-0.0156, -0.0062] |

This resolves one concrete source of the joint-model failure: treatment
history can improve factual prediction when it is dose/time-resolved and
strictly as-of. It does **not** promote the eICU joint-JEPA candidate or make
a causal claim. The MIMIC cohort has a different schema and no hospital
identifier, so it is external factual evidence for the feature contract, not
a direct replacement for the multi-hospital eICU runtime gate.

Artifacts:

- `mimiciv_observation_transition_report_full_v31_treatment_asof_strict.json`
- `mimiciv_treatment_context_accuracy_audit_hist_only_asof_strict_full_v31.json`
- `MIMICIV_TREATMENT_CONTEXT_ACCURACY_FINDINGS_HIST_ONLY_ASOF_STRICT_FULL_V31.md`
