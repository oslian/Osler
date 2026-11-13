# Four Generalization Diagnostics: Results

Date: 2026-09-03. Completed, exploratory, candidate-only.

## Decision

All four proposed directions were implemented and evaluated. None adds a target
that passes the complete masked-current reconstruction diagnostic in both
MIMIC-IV and NWICU. Do not promote these candidates or change serving weights.
The incumbent remains **41 catalog cells: 28 movable, 13 persistence**.

This is not a test of 41 independent reconstruction targets. Those cells contain
15 distinct numeric variables plus a separate hypoxemia-event task. Current
reconstruction and future forecasting remain separate capabilities.

| Question | Controlled comparison | Observed result |
|---|---|---|
| Does the readout need information lost by the encoder? | Legal raw observations/history -> ridge vs the same masked input -> frozen JEPA context -> ridge | Latent ridge has lower MAE for 15/15 eICU hospital-test targets, 15/15 MIMIC targets, 13/14 NWICU targets. The simple raw probe does not rescue generalization. |
| Was the head under-trained? | Matched 32-update head vs source-selected 32/128/512-update head, seven-seed ensemble | Longer training improves many averages, including creatinine and BUN, but hurts others and some care units. No new complete pass. |
| Can source group imbalance be repaired? | Ordinary L1 vs care-unit-balanced sampling vs mixed group-DRO risk | Some target MAEs improve, others regress. Neither weighting method adds a complete pass. |
| Can shared state be improved without damaging forecasts? | Joint body-projection/reconstruction update, with or without incumbent replay preservation | Replay reduces the number of pooled forecast regressions, but does not eliminate them. Both remain rejected for serving. |

The raw/latent probes have different feature widths, so they are **not** a
parameter-count-matched test of representation sufficiency. These results do
not prove the encoder preserves all useful information, nor that more capable
raw-input models could never help. Each probe does have its own independently
trained, matching-capacity placebo.

## Frozen Design and Data

The [protocol](../protocols/MASKED_GENERALIZATION_PROTOCOL_20260903.md) was written before
the main run. All model selection was completed using eICU tuning patients
before inspecting this run's hospital-test, MIMIC or NWICU scores.

| Source role | Patients | Hospitals |
|---|---:|---:|
| Fit | 1,736 | 152 |
| Tune | 1,737 | Same 152 |
| Hospital test | 1,288 | 55 different hospitals |

These are the 4,761 patients with usable anchors from the existing
hospital-balanced 5,000-patient extraction. Fresh-label support differs by
target; the report records it separately. Hospital IDs define heldout groups,
care-unit labels define weighting/audits, and neither is an encoder input.

One outcome-independent anchor per patient is selected for each target. Truth
must be observed within 1h; other inputs must be within 6h. The target, its
predefined near-deterministic siblings and all their histories are hidden.
Every arm uses the same eligible inputs and target masking. Training uses the
existing release scaler; reported errors are in canonical physical units.

Each evaluated task uses at most 4,000 patients, not every database row. NWICU
urine lacks compatible support, so MIMIC has 15 and NWICU 14 evaluated targets.
The hospital-test cohort has all 15 targets evaluable.

Important limits:

- The historical encoder may already have seen these source patients. The 55
  hospitals are held out from this intervention, not necessarily from its
  original pretraining.
- MIMIC and NWICU have already been inspected in earlier experiments. This is
  exploratory cross-database validation, not a new untouched lockbox.
- Seven head-training seeds are averaged, not selected on external results.
  The two shared-projection arms use one optimizer seed, 73; they do not have
  seven-training-seed replication.
- All-supported-care-unit superiority is a stricter diagnostic than the
  original aggregated-unit forecast gate. Their pass counts are not directly
  interchangeable.
- Seven patient evaluation splits overlap. Multiple target/method comparisons
  use nominal bootstrap intervals, without multiplicity correction.

## Reconstruction Results

### Full Diagnostic

A complete pass requires the existing population-center/wrong-patient tests,
seven patient-split tests and all supported care-unit tests, plus a separately
trained capacity-matched placebo. Unit support is at least 30 patients; tasks
need at least 200. Thresholds were not relaxed.

| Arm | eICU hospital test | MIMIC | NWICU |
|---|---|---|---|
| Raw ridge | None | None | None |
| Latent ridge | None | None | None |
| Latent ERM, fixed 32 updates | None | None | Creatinine only |
| Latent ERM, source-selected duration | None | None | None |
| Latent balanced | None | None | None |
| Latent mixed group DRO | None | None | None |
| Shared projection, unprotected | None | None | None |
| Shared projection, replay-protected | None | None | None |

The NWICU fixed-32 creatinine pass reproduces the earlier limited positive
result; it is not a newly promoted future cell or a cross-database full pass.

### Training Duration Helps Some Targets, Not All

Source-selected ordinary ERM chose 512 updates for 8 targets, 128 for 6, and 32
for urine. Balanced training chose 512/128/32 for 10/4/1 targets; mixed DRO for
9/5/1. Selection at the maximum budget is not proof of convergence.

| Comparison | eICU lower MAE / evaluated | MIMIC | NWICU |
|---|---:|---:|---:|
| Source-selected ERM vs fixed 32 | 11/15 | 10/15 | 9/14 |
| Balanced vs source-selected ERM | 4/15 | 4/15 | 3/14 |
| Mixed DRO vs source-selected ERM | 4/15 | 7/15 | 5/14 |

Counts above are point estimates, not full gate passes. For selected ERM vs
fixed 32, the pooled paired-bootstrap intervals favor the candidate for 7,
9 and 8 targets respectively; this still does not establish every-unit benefit.

Examples of **current reconstruction MAE**, not forecast MAE:

| Target | MIMIC fixed 32 | MIMIC selected ERM | NWICU fixed 32 | NWICU selected ERM |
|---|---:|---:|---:|---:|
| Creatinine (mg/dL) | 0.729 | 0.685 | 0.710 | 0.657 |
| BUN (mg/dL) | 12.39 | 12.03 | 12.80 | 12.13 |
| Glucose (mg/dL) | 35.42 | 38.76 | 39.13 | 39.75 |
| Potassium (mmol/L) | 0.441 | 0.459 | 0.411 | 0.419 |

### Why Better Averages Still Fail

For selected-ERM creatinine in MIMIC, pooled MAE is better than the separately
trained placebo: delta -0.098, 95% CI [-0.126, -0.073]. But **Neuro Stepdown**
has delta +0.169, CI [+0.063, +0.278], across 52 patients. That is an observed
subgroup regression, not merely an interval missing significance by a little.

The same NWICU model has a favorable pooled delta of -0.102, CI
[-0.119, -0.085], but IMCU (73 patients) and INPATIENT (63) have intervals
crossing zero. These are inconclusive subgroups, not established harm. We must
not collapse those two different failure types into "just insufficient N."

For MIMIC glucose, longer source-selected training is worse even at the pooled
level than trained placebo: delta +2.932 mg/dL, CI [+2.286, +3.610]. A model
chosen on source-domain performance can learn a relation that does not transfer.
This experiment does not isolate whether the remaining shift is due to patient
case mix, measurement conventions, treatment context or unmeasured physiology.

## Forecast Retention

Only the final shared-body projection and reconstruction heads were updated.
The upstream history encoder, organ mixers and future heads were frozen. This
is not full end-to-end JEPA retraining. The protected arm adds fit-patient replay
of incumbent unmasked values, context and predicted target-future latent, with
weight 10. Replay contains no future truth or future treatment.

Real protected/unprotected models both selected 64 updates per target;
their separately trained placebo models selected 32. Selection used source
tuning only. The protected/unprotected real-model contrast therefore has equal
capacity and update budget.

All 41 catalog cells were accounted for in each database. MIMIC provides 38
evaluable numeric cells; NWICU 37. Two hypoxemia cells need separate event
labels; unused MIMIC platelets@1h support is insufficient; NWICU urine@1h/3h
has no compatible label. These are explicitly untested, not counted as passes.
Known MIMIC multisource-head fitting patients were excluded using hash-checked
compatible caches. The existing source choice for each cell was frozen.

| Database | Evaluable cells | Pooled harm without replay | Pooled harm with replay |
|---|---:|---:|---:|
| MIMIC | 38 | 11 | 3 |
| NWICU | 37 | 11 | 8 |

"Pooled harm" means the lower endpoint of the paired-bootstrap MAE-difference
CI is above zero, relative to the incumbent. Fewer harmful cells is not proof of
non-inferiority for all other cells; some harmful cells also differ between arms.

Protected cells still showing pooled harm:

- MIMIC: urine_output@1h, chloride@12h, map@24h.
- NWICU: glucose@3h/12h, potassium@12h/24h, respiratory_rate@12h,
  bicarbonate@24h, heart_rate@24h, map@24h.

For example, MIMIC MAP@24h degradation falls from +1.100 mmHg MAE without
replay to +0.186 with replay. NWICU falls from +1.368 to +0.032. Both protected
intervals remain strictly above zero. Preservation helps, but is insufficient.

The frozen-encoder reconstruction probes do not alter ordinary forecasts by
construction. No changed representation is attached to the old conformal
calibrator. This run does not claim newly calibrated intervals, multi-step
physiological consistency or latent-rank validation.

## What This Resolves

There are several different issues, not one demonstrated universal cause:

1. Training budget leaves recoverable signal in some frozen-state readouts.
2. Average improvement can coexist with subgroup harm or insufficient subgroup
   evidence; care-unit balancing and mixed DRO do not uniformly repair it.
3. Updating shared state for reconstruction can interfere with existing future
   readouts; replay reduces, but does not remove, this interference.
4. A simple legal-raw-input probe does not outperform the current representation
   broadly. "The encoder discarded everything" is not supported by this test.

Do not tune another variant on these external results and call the same data
untouched validation. A further experiment must freeze source-only selection,
separate subgroup harm from uncertainty, and reserve genuinely new validation
data. This bounded run neither solves all generalization nor proves it impossible.

## Reproduction and Artifacts

From the repository root with its existing Python environment:

```bash
python -m scripts.audit_masked_generalization
```

Paths and fixed settings are in the script and report. Re-running requires the
local licensed event tables and exclusion caches listed in the report. Use a
new `--output-dir`; the script refuses to overwrite a completed experiment.

- Aggregate result, input/code hashes and learning curves:
  `artifacts/masked_generalization_20260903/report.json` (`complete: true`).
- All frozen-probe weights: `frozen_probes.joblib` in the same directory.
- Shared candidate weights: `shared_unprotected_real.pt`,
  `shared_unprotected_placebo.pt`, `shared_protected_real.pt`,
  `shared_protected_placebo.pt` in the same directory.
- Helpers: `scripts/masked_generalization_methods.py`.
- Tests: `tests/test_masked_generalization.py`.

Artifacts are local research candidates, not release replacements. No raw
patient rows are saved in the project, and nothing from this run was pushed.
The script verifies serving artifact hashes are unchanged.

Verification completed: 65 related unit/regression tests passed;
`python -m scripts.validate_release` returned `status: ok`, 41 catalog cells,
28 movable, 13 persistence. No causal or clinical claim; no next-measurement-time
model added.
