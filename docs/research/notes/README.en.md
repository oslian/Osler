# Whole-Body JEPA Research Release

This private repository packages a unified 2026-09-02 readout selector on the
2026-08-25 frozen factual whole-body forecasting backbone: JEPA, target heads, external-validation
registry, patient CSV runtime, canonical MIMIC-IV/eICU extractors, and synthetic
examples.

The default patient API now uses one checkpoint pass per horizon and selects
base JEPA, the original shared head, or the multi-source shared head per cell.
`models/unified_source_selector.json` binds that decision and its interval to
the exact weights. Development selection, calibration and testing are patient
disjoint. NWICU checks the frozen decision; it never selects a runner-up.
Failed replacements retain the incumbent and its original evidence, not a new
claim of NWICU success. Both databases have previously been inspected.
See [the complete comparison](../../../UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md).
Counts below describe the original baseline; use the selector/report for the
current per-cell sources. These counts are overlapping, not additive.

The interval-only follow-up promotes `bicarbonate@24h` from persistence to the
frozen multi-source head with a patient-conditional interval. The current total
is 28 movable cells and 13 persistence cells. This is reused research validation,
not a new independent lockbox. See [the rescue audit](../../../PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md).

The release also ships a frozen cross-database replication layer: 27 cells
selected/trained on eICU passed patient, care-unit, and forward-time gates on
MIMIC-IV without MIMIC refitting. Fourteen cells remain explicit challenge
candidates. See [CROSS_DATABASE_REPLICATION.md](../../../CROSS_DATABASE_REPLICATION.md)
and run `whole-body-lockbox` on a scored table from a new database.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
whole-body-validate
whole-body-predict --input examples/patient_1.csv --output outputs/patient_1.json
```

The runtime accepts wide or `variable,value` longitudinal CSV. Blank means
missing; physical zero remains observed. With multiple rows, include
`hours_since_onset` or a parseable timestamp. Input units and all 41 cell
statuses are documented in the main [Chinese README](../../../README.md).

Missing current values and future forecasts are separate objects. If
bicarbonate or heart rate has never been measured and at least four related
physiological inputs are available, `current_estimate` contains an eICU-trained,
MIMIC-IV-gated cross-target nowcast with a calibrated 90% interval. Observations
always take precedence, unvalidated targets remain missing, and a nowcast is
not fed into future forecasting without a separate cascade-error audit.

The release contains 41 forecast cells. Twenty-seven passed the eICU-to-MIMIC
external point gate; fourteen use persistence. A validated cell also requires
an observed target history for patient-level inference. Otherwise it returns
missing rather than a fabricated number.

Twenty-four of the twenty-five continuous movable cells also emit release-bound
90% conformal intervals. They passed disjoint-patient, care-unit, and
forward-time coverage audits on MIMIC-IV. One continuous cell remains
uncalibrated. The 1h and 3h hypoxemia cells use separate calibrated event-risk
heads and emit probabilities in `[0, 1]`; they are no longer obtained by
thresholding a continuous O2Sat forecast. Missing or stale O2Sat causes explicit
abstention.

This is research software, not a medical device. It makes factual forecasts
only. It does not estimate treatment effects, and every output has
`causal_claim_allowed: false`.

## Candidate physiological subtype layer

The repository also contains a candidate-only soft subtype adapter. It uses
only pre-anchor shared context, patient belief, and measurement-pure state.
Patients with few observations borrow a residual prior from similar discovery
patients; resolved personal errors gradually take over through hierarchical
shrinkage. The formal 2026-08-24 eICU-to-MIMIC-IV seven-seed test promoted
0/39 continuous cells, so it does not modify the 41-cell runtime. `MAP@12h`
and `respiratory_rate@6h` were credible near-misses but remain candidate-only.
See [PHYSIOLOGY_SUBTYPE_LAYER.md](PHYSIOLOGY_SUBTYPE_LAYER.md).

Development-only eICU follow-ups tested causal patient trajectories,
three-model consensus, target-specific history windows, and hospital-balanced
residual priors. No cell passed every patient, unseen-hospital, and forward-time
gate. MIMIC-IV was not reused for tuning, and the release runtime remains
unchanged.

A further stable-background ablation tested age, sex, BMI, pre-anchor
comorbidities, causal history statistics, separate multi-view embeddings, and
target-specific ridge residuals. `MAP@6h` retained a local signal but passed
0/3 unseen-hospital splits, so no background adapter entered the runtime.

Final eICU-only ablations added strictly pre-anchor `hist_*` treatment
context, discovery-hospital worst-case stacking, and a three-member direction
consensus gate. `MAP@6h` still passed only 5/7 patient, 0/3 unseen-hospital,
and 1/1 forward-time splits. The robust selector chose zero correction for
every cell. Future `act_*` fields are rejected, and the runtime is unchanged.

A real cohort-selection defect was then removed. The old disease-name source
preference restricted MAP development to 12 hospitals; canonical source
selection now uses strict future-label hospital depth and patient support.
With a patient-dense 6,000-row sample, `MAP@12h` reached 5/7 patient and 2/3
unseen-hospital splits, while respiratory rate reached 3/3 hospital splits.
Neither passed the complete patient/hospital/time gate. Cross-fitted selective
gating, consensus, robust stacking, and causal online patient residual updates
did not close the remaining gap, so the runtime is unchanged.

A subsequent audit found that history summaries had been computed after row
sampling and that separate ICU stays could share a patient-relative clock.
History is now built from the complete pre-anchor stay, while patient identity
remains the split unit. This repaired `MAP@12h` forward-time transfer (1/1),
but patient and hospital stability remained 2/7 and 1/3. Complete event-state
and direct target-ridge variants also failed the full gate, so unobserved
future acute events remain abstained rather than fabricated as point residuals.

The weights are derived from credentialed MIMIC-IV/eICU data and must not be
made public or shared outside the applicable PhysioNet agreements. Keep this
repository private; see [NOTICE.md](NOTICE.md).

## Frozen release masking audit (2026-09-03)

All 41 catalog cells were inventoried. The packaged checkpoint, not a separately
trained reconstruction candidate, was evaluated after removing the target,
predefined siblings, and their entire input history. Every latent/readout was
recomputed, including base point and repeated target features.

Same-anchor reconstruction evaluated 15 MIMIC and 14 NWICU variables, with zero
complete diagnostic passes. Hematocrit and platelets passed seven patient
splits in both databases but not every supported care unit. Missing-target
forecasting evaluated 38 MIMIC and 37 NWICU cells; direct JEPA creatinine@12h
passed only in MIMIC. Event labels, NWICU urine, and unused MIMIC platelets@1h
support remain separate limitations, not failed predictions.

These bounded tests use at most 4,000 patients per task, one selected anchor
each, and same-model wrong-patient context controls. The all-supported-unit
gate is stricter than the existing aggregated-unit serving gate. No new
promotion, missing-input interval, or claim of complete physiological
understanding follows. Normal observed-anchor forecasting remains unchanged.
See [the full audit](../../../RELEASE_MASKED_CELLS_FINDINGS_20260903.md).

### Matched-mask repair and readout ensemble

The follow-up corrected an anchor/future-latent reconstruction mismatch and
trained A/B/C0/C controls on eICU. Frozen-encoder readout training improves many
masked targets without changing ordinary forecasts; joint encoder updates
improve some nowcasts but regress several incumbent forecasts. A seven-seed
readout ensemble with an independently trained equal-capacity placebo passes
the complete NWICU creatinine reconstruction diagnostic, but not every MIMIC
care unit. No target passes the complete screen in both databases, and no
runtime cells were promoted. See [the methods and results](../findings/MASKED_SHARED_STATE_REPAIR_FINDINGS_20260903.md).

### Source cohort repair

The 5,000-stay eICU source was an ID-prefix sample covering only 12 hospitals.
Patient-first, hospital-round-robin sampling now selects 5,000 patients across
208 hospitals (207 with usable anchors). With the same frozen encoder and
readout optimization budget, reconstruction MAE improves for 10/15 MIMIC and
11/14 NWICU targets, but no additional target clears the complete diagnostic.
Patient count, care-unit mix and the rebuilt ledger also change; this does not
isolate hospital sampling as the sole cause. Runtime remains 28 movable cells
and 13 persistence cells. See [results and failure breakdown](../findings/MASKED_SOURCE_COHORT_REPAIR_FINDINGS_20260903.md).

### Four controlled generalization diagnostics

Raw/latent ridge probes, longer head fitting, care-unit balancing/group DRO,
and limited shared-projection updates with/without forecast replay were tested.
Selection used eICU fit/tune only, reserving 55 source hospitals. MIMIC and NWICU
remain previously inspected external validation, not new untouched lockboxes.
Longer training lowers current-reconstruction MAE for 10/15 MIMIC and 9/14 NWICU
targets, but adds no complete cross-database pass. Replay reduces pooled
forecast-regression counts from 11 to 3 in MIMIC and 11 to 8 in NWICU, without
eliminating interference. Serving weights and 28 movable / 13 persistence cells
are unchanged. See [methods, results and limits](../findings/MASKED_GENERALIZATION_FINDINGS_20260903.md).

### Robust and nonlinear frozen-state readouts

Median regression, Huber regression and small absolute-error boosted trees were
tested with source-only selection and seven matched real/placebo bootstrap bags.
Trees improve pooled current-reconstruction MAE over the previous selected
neural head for 9/15 MIMIC and 8/14 NWICU targets, but add no complete diagnostic
passes. Tail errors and significant care-unit placebo failures remain despite
average Hgb/Hct improvements. No serving weights changed; the 28/13 split is
unchanged. See [results and tail diagnostics](../findings/MASKED_ROBUST_READOUT_FINDINGS_20260903.md).
