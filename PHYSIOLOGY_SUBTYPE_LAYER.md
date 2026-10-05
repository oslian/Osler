# Physiological Subtype Layer

## Purpose

This candidate layer lets a patient with only a few observations borrow a
forecast residual prior from physiologically similar discovery-cohort
patients. It does not replace the frozen whole-body JEPA and it is not a
diagnosis classifier.

```text
validated whole-body JEPA point
            +
soft physiological-subtype residual prior
            +
resolved personal-error update
            =
candidate personalized point
```

## What defines a subtype

The router uses only three pre-anchor blocks from the trained 311-value
contract:

- 64-value shared whole-body context;
- 96-value patient belief state;
- 53-value measurement-pure physiological state.

It excludes target-future latent, target-organ latent, base forecast, current
target duplication, future observations, hospital ID and care-unit ID. The
213 selected values are fitted with discovery-only robust scaling and PCA,
then a Bayesian Gaussian mixture returns a soft posterior over subtypes.

A patient is therefore represented as probabilities, not a hard label. A
result may look like `0.62 subtype 2 + 0.27 subtype 5 + 0.11 subtype 1`.
Names such as "older diabetic subtype" may only be attached after an
independent profile audit shows that age and diabetes are enriched in that
component. The model itself must not invent that clinical label.

## Few-observation shrinkage

For every target-horizon cell, discovery patients estimate one residual prior
per subtype. Each prior is shrunk toward zero and disabled when its effective
patient support is below the configured minimum.

For a new patient with no resolved forecast errors:

```text
candidate = frozen JEPA point + probability-weighted subtype residual
```

After the patient's earlier forecasts have reached their outcome time, their
resolved errors can update the correction:

```text
patient_weight = n / (n + patient_prior_strength)
correction = (1 - patient_weight) * subtype_prior
           + patient_weight * mean(resolved_personal_errors)
```

Unresolved outcomes must never be used. With the default prior strength of 4,
one resolved error contributes 20%, four contribute 50%, and sixteen
contribute 80% of the correction.

## Validation boundary

The implementation in `whole_body_jepa/subtypes.py` is candidate-only. It is
not connected to the 41 promoted runtime cells. Each exact cell must pass:

1. seven patient-heldout seeds;
2. hospital, care-unit and temporal heldout;
3. the frozen population model;
4. an equal-capacity shuffled-subtype placebo;
5. a no-subtype ablation;
6. patient-level bootstrap and conformal coverage;
7. subtype assignment stability and minimum effective patient support.

Only a validated registry may enable a subtype correction. Failed cells keep
the current model or persistence. Subtype membership is an internal model
state, not a diagnosis, biological truth, treatment effect or causal claim.

## Minimal API

```python
from whole_body_jepa import SoftPhysiologySubtypeModel

model = SoftPhysiologySubtypeModel(subtype_count=6).fit(
    features_311,
    base_points=base_forecasts,
    observed_targets=future_values,
    observed_mask=future_observed,
    subject_ids=subject_ids,
    cell_keys=cell_keys,
)

candidate = model.candidate_predict(
    one_feature_311,
    cell="glucose@6h",
    base_point=base_point,
    resolved_patient_errors=[-8.0, -5.0],
)
```

The fit rows must be discovery-only. Evaluation and private-patient rows must
not be included in subtype discovery or residual-prior fitting.

## Formal external result (2026-08-24)

The formal gate fitted the shared subtype router and cell residual priors on
eICU only, then opened MIMIC-IV once for external scoring. It evaluated 39
continuous target-horizon cells with seven initialization seeds. Every seed
was compared with both the current release point and a patient-level
shuffled-responsibility placebo with identical model capacity.

**Result: 0/39 cells were eligible for runtime integration.** The release
registry was not changed.

The negative result was not flat:

- `map@12h` improved MAE from 10.6584 to 10.5357. Its seed-ensemble passed
  patient, care-unit, and forward-time confidence intervals against both base
  and placebo, but only 6/7 individual initializations beat both comparators.
- `respiratory_rate@6h` improved MAE from 3.8283 to 3.7654 and passed all
  seven seeds plus all three gates against base. Against placebo, the
  care-unit CI was `[-0.1689, 0.0019]`, so it narrowly failed.
- `map@6h` passed 7/7 seed point comparisons and the patient gate, but its
  care-unit and forward-time intervals crossed zero against base.

These near-misses show that physiological stratification contains some
transportable perfusion and respiratory signal. They do not prove a stable
runtime gain: the gain remains sensitive to mixture initialization or care
unit composition. MIMIC-IV labels must not now be reused to tune subtype
count, shrinkage, or cell selection; a revised method requires a new external
lockbox.

The complete machine-readable result is
`artifacts/subtype_external_validation/subtype_external_validation.json`.

## eICU development follow-up (2026-08-24)

MIMIC-IV was not read again after the external lockbox was opened. Three
development-only eICU experiments tested whether the near-misses were caused
by noisy single-row subtype assignments or discovery-hospital imbalance:

1. a causal 48-hour within-patient trajectory median plus a three-member
   prediction ensemble;
2. target-specific history windows (`MAP@6h`: 6 hours, `MAP@12h`: 12 hours,
   `respiratory_rate@6h`: 6 hours) blended with the current state;
3. hospital-balanced residual-prior fitting, where each discovery hospital
   and each patient membership within that hospital receive equal total
   weight. Hospital identity remains excluded from model inputs.

All three variants were rejected. Target-specific windows improved patient
split stability relative to the generic 48-hour median, but none passed all
seven patient splits, all three unseen-hospital splits, and the forward-time
split. Hospital balancing also failed to rescue a cell:

| Cell | Patient passes | Hospital passes | Time passes | Median patient delta vs base | Result |
|---|---:|---:|---:|---:|---|
| `map@6h` | 2/7 | 0/3 | 1/1 | -0.0050 | rejected |
| `map@12h` | 2/7 | 1/3 | 0/1 | +0.0080 | rejected |
| `respiratory_rate@6h` | 4/7 | 1/3 | 1/1 | -0.0148 | rejected |

Negative delta favors the candidate. The respiratory-rate signal is the most
repeatable, but one unseen-hospital split still loses to the base or row-level
subtype model. The result rules out simple temporal smoothing, initialization
ensembling, target-specific history length, and hospital reweighting as
sufficient fixes. Residual corrections vary by patient cohort, hospital, and
time, so this subtype layer remains research-only and the release runtime is
unchanged.

Machine-readable development artifacts:

- `trajectory_consensus_eicu_audit.json`: generic 48-hour median;
- `target_specific_trajectory_consensus_eicu_audit.json`: target-specific
  windows;
- `hospital_balanced_target_trajectory_eicu_audit.json`: final balanced audit.

## Stable-background phenotype ablation (2026-08-24)

A second eICU-only development series tested the remaining untried proposal:
stable patient background. The candidate used age, sex, admission BMI, and
diabetes, CKD, heart-failure, and COPD documentation only when it was known by
the anchor. A separate causal history branch summarized each measurement's
trailing level, slope, median absolute deviation, observation density, and
measurement age. No hospital or care-unit identifier entered either branch.

Four controlled variants were evaluated:

| Variant | Best result | Outcome |
|---|---|---|
| stable background only | `map@6h`: patient 5/7, hospital 0/3, time 1/1 | rejected |
| history summary only | every cell 0 patient passes | rejected |
| stable + history combined | every cell 0 patient passes | rejected |
| separate multi-view embeddings + subtype prior | `map@6h`: patient 4/7, hospital 0/3, time 1/1 | rejected |
| separate multi-view embeddings + target ridge | higher error than base for all median patient results | rejected |

The first combined implementation exposed a real engineering flaw: a small
background vector was being compressed in the same PCA as 213 latent values.
The implementation now scales and embeds the latent and background views
separately before fusion. A synthetic regression test confirms that a
continuous phenotype signal survives this path. Nevertheless, the corrected
real-data experiment still failed unseen-hospital transfer.

This narrows the scientific result. Stable background contains a local
`MAP@6h` signal, but age, sex, BMI, common comorbidity labels, history
statistics, consensus mixtures, and continuous ridge residuals are not enough
to make that correction transportable. The remaining variation is more likely
acute phenotype-by-hospital interaction or unobserved care context than a
missing static subtype label. No cell was promoted and the runtime remains
unchanged.

Additional artifacts:

- `stable_only_phenotype_eicu_audit.json`;
- `history_only_phenotype_eicu_audit.json`;
- `stable_background_phenotype_eicu_audit.json`;
- `stable_multiview_subtype_prior_eicu_audit.json`;
- `stable_multiview_ridge_eicu_audit.json`.

## Treatment-context and robustness ablations (2026-08-24)

A final eICU-only series tested whether the unstable subtype correction was
missing known pre-anchor treatment context or was being driven by disagreeing
ensemble members. Treatment covariates were restricted to `hist_*` fields.
The validator rejects every `act_*` field, so no future treatment enters the
phenotype or forecast.

| Variant | `map@6h` patient | hospital | time | Outcome |
|---|---:|---:|---:|---|
| treatment history only | 5/7 | 0/3 | 0/1 | rejected |
| stable background + treatment history | 5/7 | 0/3 | 1/1 | rejected |
| discovery-hospital worst-case stack | 0/7 | 0/3 | 0/1 | selected zero correction |
| stable + treatment, consensus correction | 5/7 | 0/3 | 1/1 | rejected |

Treatment history therefore contains a real local `MAP@6h` signal, but it is
not transportable. In unseen hospitals the candidate usually improves over
the frozen base and trajectory prior, yet loses to the simpler row-level
subtype correction. Different hospitals prefer different correction sources.

The group-robust stack searched convex combinations and correction scales
using discovery hospitals only. It selected correction scale zero in every
cell: under a worst-hospital objective, the safest discovery-time decision is
the frozen base model. This is an informative fail-closed result rather than a
runtime improvement.

The consensus gate applied a correction only when all three independently
fitted subtype models agreed on direction and the mean correction exceeded its
between-model standard deviation. It did not change the conclusion:
`map@12h` passed 1/7 patient splits and `respiratory_rate@6h` passed 1/7;
neither passed the forward-time gate.

Together these ablations rule out missing stable background, simple causal
history summaries, pre-anchor treatment history, ensemble sign disagreement,
and discovery-hospital robust blending as sufficient explanations. The
remaining residual is not reliably identifiable from the available
pre-anchor covariates across hospitals and time. No subtype variant is promoted,
and the production runtime remains unchanged.

Additional artifacts:

- `treatment_multiview_subtype_prior_eicu_audit.json`;
- `stable_treatment_multiview_subtype_prior_eicu_audit.json`;
- `stable_treatment_group_robust_stack_eicu_audit.json`;
- `stable_treatment_consensus_gate_eicu_audit.json`.

## Canonical discovery-source and patient-update audit (2026-08-24)

The follow-up found and removed a real cohort-selection defect. The eICU
loader previously chose a disease-named preferred module for each target. For
MAP this selected a cardiovascular file with only 12 hospitals, despite strict
MAP labels being available in more than 200 hospitals in the AKI and sepsis
transition sources. Source selection now ranks every compatible parquet by
strict `exact/near`, positive-weight label support: hospitals with at least ten
patients are no longer allowed to outweigh broad transfer coverage. The final
ranking is hospitals with at least three strict-label patients, hospitals with
at least ten, hospitals with at least five, total patients, then rows.
Only columns required by JEPA inference and pre-anchor treatment context are
loaded.

The audit also changed sampling from at most ten to at most two rows per
patient. The fixed 6,000-row budget therefore represents roughly 3,000
patients instead of roughly 600, providing enough depth for unseen-hospital
testing. MIMIC-IV was not read.

On this corrected cohort, the ungated `MAP@12h` subtype candidate improved to
5/7 patient splits and 2/3 unseen-hospital splits. The respiratory-rate
candidate passed 3/3 unseen-hospital splits. Neither completed the full gate:

| Cell | Patient | Hospital | Time | Outcome |
|---|---:|---:|---:|---|
| `map@6h` | 1/7 | 0/3 | 0/1 | rejected |
| `map@12h` | 5/7 | 2/3 | 0/1 | near-miss, rejected |
| `respiratory_rate@6h` | 0/7 | 3/3 | 0/1 | rejected |

A hospital-cross-fitted selective gate used only out-of-fold discovery-hospital
predictions. With the broader patient sample it disabled every correction in
all 11 outer splits, because no threshold improved the worst discovery
hospital. A target-only three-member consensus fixed `MAP@12h` forward-time
performance (1/1), but reduced patient and hospital stability to 1/7 and 0/3.
Discovery-hospital robust stacking selected zero correction in every split.

Finally, a causal online patient update was tested. A resolved error could
enter the patient state only after its actual future endpoint. It worsened the
late trajectory: prior frozen-base errors were not a stable patient-specific
bias. This rules out simple accumulating residual means as the missing
personalization mechanism.

The source-selection bug is fixed, but no subtype cell is promoted. The
remaining failure is now sharply localized: the residual direction changes
across patient samples and time, while the available pre-anchor phenotype does
not identify that change. Runtime remains unchanged.

Machine-readable artifacts:

- `canonical_source_patient_dense_ungated_eicu_audit.json` (final canonical
  source rerun; authoritative ungated result);
- `broad_patient_dense_ungated_eicu_audit.json`;
- `broad_patient_dense_crossfit_gate_eicu_audit.json`;
- `broad_patient_dense_personal_update_eicu_audit.json`;
- `map12_broad_consensus_eicu_audit.json`;
- `map12_broad_group_robust_eicu_audit.json`.

## Complete-stay pre-anchor event-state repair (2026-08-24)

The pre-anchor history path contained two concrete extraction defects. History
summaries were calculated after patient-balanced sampling, so at most two
retained rows were treated as the complete trajectory. Acute trajectory
aggregation also used the patient identifier with a stay-relative clock,
which could merge separate ICU stays from the same patient.

The repaired path now computes history from every available row in the
selected ICU stay before anchor sampling. Patient identifiers remain the split
unit, while stay identifiers are the acute trajectory unit. A new
`event_state` view combines stable background, complete pre-anchor physiology,
measurement ages, and strictly pre-anchor treatment history. No future
`act_*` field is accepted.

The repair improved temporal validity but did not produce a promotable cell:

| MAP@12h method | Patient | Hospital | Time | Outcome |
|---|---:|---:|---:|---|
| stay-safe stable+treatment prior | 2/7 | 1/3 | 1/1 | rejected |
| complete event-state prior | 1/7 | 1/3 | 1/1 | rejected |
| complete event-state ridge | 1/7 | 0/3 | 0/1 | rejected |
| uncompressed direct event-state ridge | 1/7 | 1/3 | 0/1 | rejected |

The direct residual heads over-corrected, especially in the forward-time
split. The supported subtype prior was safer, but did not consistently beat
the trajectory-only and row-level controls. The extraction defects are fixed;
the remaining point residual depends on future acute events or care changes
that are not identified by the available pre-anchor state. The safe response
is still abstention/persistence rather than a fabricated correction.

Machine-readable artifacts:

- `map12_stay_safe_stable_treatment_prior_eicu_audit.json`;
- `map12_complete_stay_event_state_prior_eicu_audit.json`;
- `complete_stay_event_state_ridge_eicu_audit.json`;
- `map12_complete_stay_event_state_direct_ridge_eicu_audit.json`.

## Pre-anchor clinical-event observability audit (2026-08-24)

The complete-stay state was extended with strictly pre-anchor clinical events
from eICU treatment, care-plan, nursing-assessment, physical-exam, and
respiratory-charting tables. The event categories were hemodynamic concern,
respiratory support, infection concern, renal support, neurologic concern, and
fluid resuscitation. Each event becomes available only at
`max(event_offset, entry_offset)`, preventing a late-entered record from being
treated as information known earlier.

This repaired the observability question but did not make the future residual
transportable:

| Cell and candidate | Patient | Hospital | Time | Outcome |
|---|---:|---:|---:|---|
| `MAP@12h`, clinical-event state | 2/7 | 0/3 | 1/1 | rejected |

The candidate usually improved over the frozen base and capacity-matched
placebo, and it passed the forward-time split. It did not consistently beat
the physiology-trajectory and row-level controls across unseen patients and
hospitals. The remaining error therefore cannot be assigned a unique point
correction from the available anchor-time evidence.

The runtime policy is fail-closed: clinical events are observation-only until
external validation passes; unobserved future events are never imputed as
facts; and the output must retain an already validated fallback or abstain.
This fixes unsafe output behavior without pretending that a future acute event
has become identifiable.

Machine-readable artifact:

- `map12_complete_stay_clinical_event_prior_eicu_audit.json`.
