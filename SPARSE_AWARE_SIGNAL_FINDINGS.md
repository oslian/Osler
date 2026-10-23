# Sparse Signal Findings

Date: 2026-07-13

## Question

Does a sparse target contain useful information because its value is stable, or
is the apparent failure caused by the observation process and extraction noise?

## Candidate tested

`SparseAwarePureJEPABodyModel` uses the same measurement-pure history encoder
as the pure-JEPA candidate, but adds independent target heads for:

1. future value conditional on an observed future value;
2. whether the target is observed at the requested horizon;
3. a target-specific uncertainty scale;
4. probability of remaining within a normalized stable band.

The observation-process head was compared with a prevalence-only baseline. The
value head was compared with persistence. This is a diagnostic candidate, not
a promotion decision.

## Audit result

The audit ran across all 15 eICU body-system modules, 239 supported target
cells, a patient-held-out split, six horizons, one training epoch, and a
1,000-stay-per-horizon cap.

- 158/239 target cells had observation rate below 25%.
- 115/158 sparse cells had a lower point-estimate MAE than persistence in this
  single candidate split. This is not a significance or promotion result.
- 69/239 target cells beat the prevalence-only measurement-process baseline on
  Brier score. The measurement-process head is useful for some targets, but
  not universally.
- After target-specific robust median/MAD/IQR scaling and a separate
  fit/calibration/test split, 186/239 target cells passed the nominal
  0.87--0.93 split-conformal coverage band; 47 were rejected and 6 lacked
  calibration support. These are candidate calibration results, not a
  seven-seed hospital-held-out promotion result.
- The stable-state head did not reliably beat its prevalence baseline on the
  sparse subset. Stability is therefore not yet a validated standalone
  signal.

## Interpretation

Sparse measurement is not equivalent to no signal. The data support three
different signals:

- the target value can be stable, making persistence the correct point
  forecast;
- the decision to measure a target is informative and can sometimes be
  predicted;
- a target may have a useful current-state estimate even when its future
  change is not predictable.

However, the current audit does not show that sparse future values are
reliably forecastable. The first version produced overly wide intervals
because global standard-deviation scaling and the raw uncertainty head were
too conservative. Robust target scaling plus split-conformal calibration
improved interval construction, but value and measurement heads still need
the full promotion gate.

## What changed in runtime

The sparse-aware checkpoints are saved in `sparse_aware_checkpoints/` and can
be loaded by `pure_jepa_runtime.py`. Runtime output now keeps these fields
separate:

- `point`: factual candidate or persistence, depending on the promotion
  registry;
- `observation_probability`: probability that the target is observed at the
  requested horizon;
- `stable_probability`: descriptive stability estimate;
- `uncertainty_scale`: raw candidate scale;
- `interval_status`: `calibrated` only when both checkpoint and registry are
  validated and the target has a conformal PASS; otherwise it remains
  `needs_runtime_calibration_artifact`.

The registry still controls movement. A sparse target cannot move away from
persistence merely because the sparse-aware model produced a number.

## Remaining causes to test

The most likely remaining limitations are:

1. the observation process is informative but only partially represented by
   mask and measurement age;
2. a fixed horizon is not the same as the next measurement time;
3. target-specific units, timestamp alignment, and extraction noise dilute
   sparse signals;
4. hidden drivers such as medication, procedure, organ reserve, or disease
   severity are not fully observed;
5. one epoch and a capped cohort are insufficient for a final target gate.

The next valid audit is therefore not to force sparse targets to move. It is
to add next-measurement-time labels, audit timestamp/unit provenance, and fit
target-specific conformal residuals on patient- and hospital-held-out data.

## Full promotion audit

The complete sparse-aware gate was run across all 15 modules using seven
patient-held-out seeds, hospital-heldout evaluation, and conformal evaluation
for every patient seed. The controlled audit budget was one epoch and at most
1,000 stays per horizon, so it tests the promotion protocol but is not a
full-data final training run.

The number of target cells passing all three evidence gates was:

| Module | Full-gate target cells |
| --- | ---: |
| AKI | 8 / 14 |
| Acute neuro | 4 / 12 |
| Cardiac injury | 3 / 18 |
| Cardiovascular instability | 5 / 13 |
| Coagulopathy/heme | 2 / 18 |
| Electrolyte/acid-base | 2 / 17 |
| Endocrine stress | 3 / 16 |
| GI/pancreatic/nutrition | 2 / 20 |
| Hepatic failure | 2 / 14 |
| Immune/inflammatory | 4 / 16 |
| Integumentary/skin wound | 4 / 18 |
| Musculoskeletal/rhabdo | 1 / 16 |
| Respiratory | 6 / 15 |
| Sepsis | 11 / 18 |
| Toxic/metabolic | 4 / 19 |

Total: 61 target cells passed the evidence gates. The new
`sparse_aware_gate_registry.json` deliberately sets `can_move: false` for all
of them and records `promotion_ready_checkpoint_retrain_required`. This is
because the saved checkpoints were trained in the controlled candidate run,
not after the gate on the full cohort. Runtime will therefore continue to use
persistence until a full-data retrain and a final artifact audit are complete.

## Final full-cohort retrain

The requested final step is now complete. All 15 modules were retrained using
the complete available cohort (`max_stays_per_horizon = None`, two epochs),
and the 61 gate-approved target cells were recorded in their module-specific
`validated_targets` lists. The artifact audit returned `all_artifacts_ok: true`
for all 15 checkpoints:

- checkpoint loads successfully;
- measurement-pure architecture is present;
- no hand-built belief input is declared;
- checkpoint target lists match the validated registry;
- every validated target has a finite conformal q90 artifact.

The research-only registry is
`sparse_aware_validated_registry.json`. It allows only those 61 target cells
to move and keeps every other target at persistence. It does not grant causal
or clinical authority. Runtime also clamps lower bounds of explicitly
non-negative laboratory quantities to zero, preventing physically impossible
interval endpoints.

## Boundary

This result is factual and observational only. It does not estimate treatment
effects, does not enable clinical promotion, and does not convert a predicted
measurement into a measured value.

## Strict final audit correction

The first final retrain exposed two statistical weaknesses in the earlier
artifact: pooled conformal q90 values and no independent evidence after the
full fit. Both are now corrected.

- Conformal calibration is stored and consumed per `target x horizon`; a
  horizon without a PASS artifact returns `needs_horizon_calibration_artifact`.
- The complete gate now audits the observation-process and stability heads
  separately against both a training prevalence baseline and a
  capacity-matched feature-permutation placebo. These auxiliary heads do not
  silently promote a value forecast.
- A separate independent hospital-heldout retraining audit was run on the
  complete available cohort. It trained on non-heldout hospitals, calibrated
  inside the training hospitals, and evaluated only on the heldout hospital.
- The strict value gate retained 17 target cells. The independent holdout
  evidence retained 13 of those for research-only runtime movement. The other
  four remain persistence fallback.

The final registry therefore allows movement only for those 13 target cells,
with causal and clinical authority disabled. The full-fit checkpoints are
research artifacts; the independent audit is the performance evidence, not an
in-sample score of the full-fit checkpoint.

The remaining boundary is structural rather than hidden: the 15 modules still
use independently trained encoders. `whole_body_forecast_contract.py` now
provides one fail-closed query object, but it explicitly reports
`shared_latent_validated: false` until a shared-encoder model is trained and
passes the same gates.

## Shared latent, exact measurement time, and external database work

The next three infrastructure gaps now have concrete artifacts:

- `whole_body_shared_jepa.py` implements one measurement-pure GRU encoder with
  a module embedding and shared target heads. A candidate smoke artifact was
  trained across five modules and 6,405 rows. It remains candidate-only until
  module-balanced patient, hospital, and target-by-horizon conformal gates are
  run.
- `next_measurement_time.py` creates strict next-event labels from raw
  timestamped measurement tables. Regular transition grids are explicitly
  marked unavailable; fixed-horizon observation probability is never renamed
  as time-to-next-measurement.
- `mimiciv_sparse_aware_external_audit.py` ran a seven-seed MIMIC-IV DKA
  external recipe audit on 755 subjects and 10,785 rows. It is an independent
  database check of the architecture, not a transfer claim for eICU
  checkpoints. Hospital-heldout evidence is unavailable because the local
  MIMIC transition artifact has no hospital identifier.

The final runtime therefore exposes the new semantics while keeping the same
fail-closed policy: shared-latent movement, exact measurement-time outputs,
and cross-database promotion require their own evidence artifacts.

## Full local-data closure audit

The three previously unresolved local-data boundaries were checked against the
actual downloaded datasets rather than assumed to be unavailable.

### Raw eICU measurement time

The downloaded eICU 2.0 archive contains `lab.csv.gz`, `vitalPeriodic.csv.gz`,
`vitalAperiodic.csv.gz`, and `intakeOutput.csv.gz`. The new
`eicu_raw_measurement_time_augment.py` scans those raw offset events and writes
68 augmented transition files for 82,382 stays. Every target receives an
independent `next_measurement_time_hr_<target>` label using the strict rule
`event_offset > anchor_offset`; an event at the anchor is not counted. This is
an observation-process label, not a future clinical outcome and not a causal
feature.

### Shared latent gate

The full shared candidate was trained on 15 available eICU body-system modules,
856,763 rows, and 56 state variables. The strict result is:

- 74 target-horizon cells passed patient-heldout, care-unit-heldout, and
  target-by-horizon conformal value evidence;
- only 3 cells also passed the module-balanced requirement;
- the global shared latent therefore remains `candidate_only` and
  `shared_latent_validated: false`.

The full candidate checkpoint is `whole_body_shared_jepa_full_candidate.pt`.
This is a useful partial result, not a promotion: a shared encoder can carry
signal for selected cells, but it has not demonstrated stable whole-body
performance across modules. Runtime must continue to use the validated
per-module registry or persistence fallback.

### Full MIMIC-IV external observation audit

The downloaded MIMIC-IV v3.1 archive contains 93,224 ICU stays and produced a
640,164-row whole-ICU transition table with 61 usable targets and 17 care
units. The independent sparse-aware recipe audit found 45 of 58 supported
target summaries winning persistence in all 7 patient-heldout seeds; the
care-unit-heldout audit also retained many fast and sparse targets, with 40
target conformal intervals passing the nominal 90% coverage band.

This closes the earlier “DKA-only external check” gap at the observation layer,
but not the hospital axis: MIMIC-IV is single-center BIDMC, so
`true_hospital_heldout` remains `not_available`. The result is external
database evidence, not cross-hospital evidence. It is factual/observational
only and does not create causal or clinical authority.

## Shared latent v2/v3 closure

The first shared encoder was not accepted by assuming that a lower training
loss meant whole-body generalization. A fair comparison was run with the same
15-module eICU cohort, 7 patient seeds, hospital-heldout evaluation, and
target-by-horizon conformal checks. The original module-balanced report was
also recomputed after fixing a read-only split array and an index-alignment bug
in the hospital module audit; the corrected v1 is the only valid baseline.

| Architecture | Module pass rate | Patient value cells | Patient conformal cells | Hospital value cells | Hospital conformal cells |
|---|---:|---:|---:|---:|---:|
| v1 corrected baseline | 73.8% | 110/189 | 97/179 | 133/194 | 147/185 |
| v2 independent module heads | 67.0% | 92/189 | 96/179 | 119/194 | 140/185 |
| v3 shared heads + bounded residual adapters | **77.3%** | **112/189** | 92/179 | **134/194** | 142/185 |

v2 was rejected: complete module-specific heads over-separated the tasks and
weakened transfer. v3 keeps the shared temporal encoder and shared target
heads, adds zero-initialized bounded module residuals, and uses balanced
module-by-horizon sampling with target-balanced loss. It is the best shared
candidate on the primary value/module metrics, but conformal coverage is not
uniformly better, so the global shared latent remains fail-closed.

The independent v3 gate produced 60 validated target-horizon cells. These cells
must pass all of: seven patient-heldout value splits, seven patient-heldout
conformal checks, hospital-heldout value and conformal checks, and evidence
from at least three modules with an 80% module pass rate. The v3 checkpoint may
move only on those registry cells; every other target-horizon cell continues to
use the validated per-module registry or persistence. No causal or clinical
authority is granted.

Artifacts:

- `whole_body_shared_jepa_v3_full_candidate.pt`
- `whole_body_shared_jepa_v3_full_candidate.json`
- `whole_body_shared_jepa_gate_v3_full_registry.json`
- `whole_body_shared_jepa_v1_fixed_full.json`
- `whole_body_shared_jepa_gate_v2_full.json`

This closes the shared-latent engineering gap as far as the available eICU
evidence supports: the design now controls negative transfer and the gate now
supports validated-cell promotion instead of requiring unsupported variables to
move. It does not justify promoting one universal whole-body latent for every
target.

## Genuine joint body-state experiment

The previous shared candidates were parameter-sharing models: each training row
still came from one module table. A genuine whole-body model was therefore
added separately. `whole_body_joint_jepa.py` first selects stays represented in
at least two modules, aligns rows by the same `stay_id + t + horizon`, merges
the observations into one masked state vector, and keeps a module-presence
mask. Its GRU history encoder is followed by a variable-token Transformer, so
the latent can receive messages from observed cardiac, respiratory, renal,
electrolyte, endocrine, hepatic, and hematologic variables before any target
head runs.

The independent joint gate also masks every other body variable and compares
the full model with an own-target-only ablation. This tests whether an apparent
gain actually uses cross-system information rather than only the target's own
history.

The full 15-module joint experiment produced 16,305 aligned rows from 336
stays, with an average of 2.28 modules represented per aligned row. It passed
the complete patient-heldout, hospital-heldout, conformal, and cross-system
ablation pipeline, but produced **0 validated target-horizon cells**. The joint
candidate is therefore saved for research, but it is not connected to runtime
promotion. The negative result is informative: the previous model was not
truly joint, and a true joint model needs either more multi-system aligned
stays, better time alignment/coverage, or a stronger inductive bias before its
cross-organ signal is stable.

Artifacts:

- `whole_body_joint_jepa.py`
- `whole_body_joint_gate.py`
- `whole_body_joint_jepa_full_candidate.pt`
- `whole_body_joint_jepa_full_candidate.json`
- `whole_body_joint_gate_full.json`

The system now distinguishes three claims: shared parameters, a genuinely
joint patient latent, and validated cross-system predictive benefit. Only the
first is established broadly; the second is implemented as a candidate; the
third remains fail-closed until the joint ablation gate passes.

### Causal as-of alignment follow-up

The exact-timestamp joint alignment was identified as a data-contract
bottleneck: real ICU measurements are asynchronous, so requiring every body
system to be observed at the same timestamp discarded most usable overlap. A
causal ``asof`` alignment path was added to ``whole_body_joint_jepa.py``.

At each anchor, it takes the latest observation from each module at or before
the anchor, applies a six-hour maximum age, and increases the measurement age
by the elapsed time. It never copies a future label from an earlier source
anchor: a ``future_<target>`` value is retained only when its source transition
starts exactly at the shared anchor. ``exact`` remains available as a
reproducibility mode.

The full 10k-stay gate used the same seven patient splits, hospital-held-out
split, conformal checks, and own-target-only cross-system ablation:

| Alignment | Rows | Subjects | Mean modules per row | Fully validated forecast cells |
|---|---:|---:|---:|---:|
| exact, masked candidate | 171,378 | 3,117 | 2.34 | 0 |
| causal as-of, six-hour age limit | 988,911 | 8,963 | 2.47 | **2** |

The as-of run validated ``bicarbonate@6h`` and ``bicarbonate@12h``. Both
passed all seven patient value splits, all seven patient cross-system
ablations, hospital value, hospital cross-system ablation, and patient
conformal checks. The same run produced 34 target-horizon cells whose
cross-system ablation passed, but most failed persistence, hospital, or
conformal requirements and therefore remain candidate-only.

Artifacts:

- ``whole_body_joint_jepa.py``: ``--alignment exact|asof`` and
  ``--asof-max-age-hours``
- ``whole_body_joint_gate.py``: the same alignment controls
- ``whole_body_joint_gate_full_10k_asof.json``
- ``whole_body_joint_jepa_full_10k_asof_candidate.pt``
- ``whole_body_joint_jepa_full_10k_asof_candidate.json``

Interpretation: asynchronous alignment was a real bottleneck and fixing it
created reproducible cross-system signal, especially for bicarbonate. It did
not justify promoting one universal whole-body model. The runtime remains
fail-closed: only the two validated cells may move in a research artifact;
all other targets continue to use the existing validated per-module sources
or persistence. This remains factual observation/forecasting only, with no
causal or clinical authority.

### Bounded horizon-label audit

The exact-label contract was then compared with a separate 0.5-hour bounded
horizon window. Inputs remained strictly backward as-of; only the future label
was allowed to come from the nearest source transition within plus or minus
0.5 hours of the requested anchor. This added no future input information.

The full 10k-stay run passed five cells under the windowed label definition:
``map@3h``, ``map@6h``, ``map@12h``, ``map@24h``, and ``sodium@12h``. The exact
label run passed ``bicarbonate@6h`` and ``bicarbonate@12h`` instead. This is a
useful result, but it also exposes a remaining scientific issue: the validated
target set changes with the label-window definition. Therefore the two reports
must not be pooled or presented as one aggregate improvement. Each is a
separate factual contract with its own horizon semantics.

The bounded-window model remains research-only. It does not establish a
universal joint latent, and it does not convert a denser label into causal
evidence. A future canonical contract still needs a target-specific event-time
definition, especially for sparse laboratory variables.

Artifact:

- ``whole_body_joint_gate_full_10k_asof_label_window_05.json``

### Joint follow-up and training diagnosis

The first joint run used only 1,000 candidate stays and left 336 stays after
exact multi-module timestamp alignment. Increasing the common-stay budget to
10,000 produced 171,378 aligned rows from 3,296 stays and 3,117 subjects. The
larger cohort made the ordinary factual forecast substantially stronger: 22
target-horizon cells passed all patient value splits in the unmasked run, and
the masked-body run reached 25.

The masked-body objective was retained because it improved the actual
cross-system ablation signal: four patient cells passed all seven ablation
splits. A stronger auxiliary loss that forced every masked target into the
future loss was tested and rejected; it reduced patient value cells from 25 to
11. It remains an optional experimental weight, defaulting to zero.

Even after the 10k masked run, no cell passed every patient value,
patient-ablation, hospital, and conformal requirement simultaneously. The
current result is therefore:

- the joint architecture is real and uses aligned multi-system states;
- the larger cohort improves factual forecasting and produces near-pass
  cross-system cells;
- stable cross-organ predictive benefit is not yet demonstrated;
- the joint candidate remains research-only, while validated per-module/v3
  artifacts remain the runtime authority.

The best saved joint candidate is
`whole_body_joint_jepa_full_10k_masked_candidate.pt`, with gate artifact
`whole_body_joint_gate_full_10k_masked.json`.

### Cross-system nowcast result

The joint gate was extended with a second, separate test: hide one current
variable from the model and ask the latent to reconstruct it from the other
body systems. This is a nowcast/reconstruction claim, not a future forecast
claim. In the full 10k masked run, 36/64 target-horizon cells passed the
hospital reconstruction comparison against the median baseline, but no cell
passed all seven patient splits. Therefore the validated cross-system nowcast
count is 0 and the candidate remains fail-closed.

The final interpretation is precise: the model is genuinely connected and can
find partial cross-system signal in some hospitals, but stable patient-level
cross-organ generalization has not yet been demonstrated. It is not permitted
to replace the validated per-module/v3 runtime sources.

### Target-specific raw event-time audit

The future-label contract was audited against the augmented eICU event-time
tables, not only against the regularized transition grid. The audit covers 68
module-horizon files, 15 body-system modules, 6 horizons, 1,128
target-horizon tasks, and 52,340,648 finite future labels.

Two different findings must remain separate:

1. **Source coverage gap.** 36 target-horizon tasks have future values but no
   event-time support in the current raw augmentation source. These are
   concentrated in respiratory device settings such as FiO2, PEEP, tidal
   volume, minute ventilation, and ventilation mode. Their values come from
   ``respiratoryCharting`` rather than the lab/vital/urine source used by
   ``eicu_raw_measurement_time_augment.py``. They are not declared invalid;
   they are marked as requiring a respiratoryCharting event-time source.
2. **Short-horizon label mismatch.** 241 target-horizon tasks at ``1h`` contain
   138,114 future labels without any strictly post-anchor raw event in the
   available source. The pattern disappears at 3h, 6h, 12h, 24h, and 48h.
   This is consistent with the extractor using a fixed ``+/-2h`` nearest-value
   tolerance: for a one-hour horizon, the current anchor observation is inside
   the future-label window and can be reused as ``tp1``. That is not a valid
   future observation and must not be interpreted as predictive signal.

The scientific correction is now explicit in the eICU extractors:
``target_tolerance_for_horizon(h) = min(2h, h/2)`` (with a small positive
floor), so ``tp1`` can only use a genuinely post-anchor event. The measurement
audit artifact is ``whole_body_joint_event_time_audit.json`` and the regression
test is ``test_whole_body_joint_alignment.py``. The audit does not promote any
target, and all claims remain factual observation/forecasting only.

The correction was then applied to a fresh, separate 1h extraction for all 15
modules and re-audited:

- 252 target-horizon tasks and 8,266,765 future labels were checked;
- 0 target-horizon tasks had a future label without a post-anchor event;
- after adding ``respiratoryCharting.csv.gz`` to the raw-event reader, every
  respiratory device setting with a future label received its own event time;
- ``future_label_event_time_consistency`` is now true for every target whose
  raw source is available.

The strict artifacts are kept separate from the original cohort:
``eicu_strict_1h/``, ``eicu_strict_1h_augmented/``, and
``whole_body_joint_event_time_audit_strict_1h.json``. This is a data-contract
repair and a falsification result, not evidence that the model itself is more
accurate. Only three target-horizon tasks have no source events and no future
labels at all (``ck_mb_index``, ``troponin_t``, and ``serum_ketones``); they
are excluded rather than fabricated. Respiratory settings now satisfy the
same future-label contract as laboratory and vital-sign targets.

### Strict 1h forecast re-audit

The 15 strict 1h cohorts were passed through the same nested factual router
gate: seven patient-held-out seeds, hospital-held-out evaluation, and 300
bootstrap samples. No model architecture or source-selection rule was changed;
only the future-label contract changed.

The number of fully validated target-horizon cells increased from 13 under the
old 1h labels to 19 under the strict labels. Newly validated cells included
sepsis lactate/urine output, AKI potassium/sodium, and respiratory PaCO2/pH.
This is not evidence that the model learned new physiology; it is evidence
that invalid short-horizon labels had been suppressing real factual signal.

The aggregate comparison is stored in
``eicu_strict_1h_label_repair_forecast_comparison.json``. It remains
observational, non-causal, and research-only.

### Joint-model problem resolution (2026-07-14)

The seven whole-body joint-model problems were addressed in the dedicated
contract and gate update documented in
``WHOLE_BODY_JOINT_PROBLEM_RESOLUTION_20260714.md``. The important result is
that the stricter contract did not magically promote a universal latent. The
final corrected horizon-scaled re-audit has zero fully validated joint cells.
The model therefore now uses target-horizon promotion with fallback, rather
than all-or-nothing promotion.

The joint loader now rejects stale values per variable, supports explicit
per-variable age overrides, and defaults to the horizon-scaled age rule. A
measurement-process-neutralized mode removes masks, ages, and module presence
from the model input. The matched 1k observed and neutralized gates both had
zero fully validated cells, so measurement-process confounding is guarded but
not claimed to be the sole cause of joint instability.

The strict 1h future-label re-extraction remains clean: zero future labels
without a post-anchor event and zero inconsistent target-horizon tasks. The
multi-hop recheck found a useful but bounded result: sepsis-to-MAP-to-renal
passes for creatinine and BUN at 24h on patient and hospital comparisons, but
not for urine output and not yet through conformal interval validation. It is
therefore a research candidate, not a universal or causal pathway.

The final module-presence audit is important: module presence is derived only
from variables belonging to that module after age filtering. A same-named
variable from another module cannot mark a module present. This invalidated
the intermediate joint artifacts that reported one or two cells; the canonical
result is ``whole_body_joint_gate_full_10k_horizon_scaled_v2.json`` with
683,911 rows and zero fully validated joint cells. The corrected matched
measurement-process comparison also has zero validated cells in both observed
and neutralized modes (65,077 rows each).
