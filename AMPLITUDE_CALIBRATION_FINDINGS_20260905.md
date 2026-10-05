# Frozen Expert Amplitude Calibration: Findings

## Decision

Completed eICU development, eICU patient/hospital evaluation, and MIMIC transfer
validation. **0/5 targets pass the complete candidate gate.** Runtime remains
unchanged: 41 catalog cells, 27 moving cells, 14 persistence fallbacks.

This is a five-target, 6h readout experiment on fixed shared-JEPA features and
fixed distributional experts. It does not invalidate existing release cells,
and it is not a retraining or validation of every whole-body target.

## Implemented Repair

Old choice: use all of the expert delta, or none.

New choice: `anchor + alpha(patient_state, context, expert_evidence) * expert_delta`,
where alpha is learned in eICU and constrained to [0,1]. Absolute-error weighted
regression directly optimizes adoption under MAE; alpha is not an event-risk
probability. Zero proposals remain zero, and observed zero is not missing.

The fixed primary reads 227 features. Controls are a matched shuffled-input
model, globally constant adoption, evidence-only adoption, no-care-context
adoption, the prior binary selector, persistence, and always-expert.
The global coefficient uses the MAE-optimal weighted median, not a guessed gain.

Intervals are recalibrated separately on disjoint eICU patients. Their widths
are not multiplied by alpha. CRPS is computed from the resulting calibrated
sample distribution, unlike the previous audit's unscaled distribution score.

Protocol was written before this run:
[AMPLITUDE_CALIBRATION_PROTOCOL_20260905.md](AMPLITUDE_CALIBRATION_PROTOCOL_20260905.md).

## Data and Independence

| Target | eICU fit patients | eICU calibration patients | Patient test | Hospital test | MIMIC test |
|---|---:|---:|---:|---:|---:|
| Creatinine | 3,486 | 1,433 | 2,895 | 2,484 | 12,000 |
| BUN | 3,440 | 1,425 | 2,883 | 2,466 | 12,000 |
| Urine output | 4,375 | 1,808 | 3,575 | 2,715 | 12,000 |
| Potassium | 4,236 | 1,757 | 3,542 | 2,931 | 12,000 |
| Bicarbonate | 4,442 | 1,786 | 3,551 | 2,726 | 12,000 |

One selected anchor per patient per target. Target cohorts overlap; do not sum
the columns into unique-patient counts. Expert-fit, readout-fit, reserved
selection, calibration, patient-test and hospital-test patients are disjoint.
Patients with held-out-hospital encounters never enter development.

No new model selection was performed in this audit. Checkpoints and conformal
scales were saved before MIMIC evaluation. Fixed expert, persistence and old
binary-selector MAE reproduce the previous report within 1e-8 for every role.
Release, protocol and predecessor hashes are recorded in the report.

MIMIC has been inspected repeatedly in this research program. These are
transfer-validation results, not a new untouched external lockbox.

## Point Forecast Results

MIMIC MAE in each target's existing canonical units; lower is better.

| Target | Persistence | Fixed expert | Previous binary | Conditional amplitude | Evidence-only amplitude |
|---|---:|---:|---:|---:|---:|
| Creatinine | 0.185562 | 0.202701 | 0.195162 | 0.193685 | 0.192971 |
| BUN | 3.029250 | 3.096715 | 3.065850 | 3.058475 | 3.063077 |
| Urine output | 111.919492 | 102.943139 | 103.749974 | 104.205102 | 103.547706 |
| Potassium | 0.428897 | 0.384478 | 0.383812 | 0.383655 | 0.382879 |
| Bicarbonate | 1.911842 | 1.885986 | 1.877767 | 1.879622 | 1.877923 |

Paired patient bootstrap, conditional minus reference; negative favors repair:

| Target | Versus previous binary, delta [95% CI] | Versus persistence, delta [95% CI] |
|---|---|---|
| Creatinine | -0.001477 [-0.002421, -0.000523] | +0.008123 [+0.006253, +0.010059] |
| BUN | -0.007375 [-0.017049, +0.001351] | +0.029225 [+0.012173, +0.046948] |
| Urine output | +0.455127 [+0.034566, +0.850328] | -7.714390 [-8.605521, -6.906652] |
| Potassium | -0.000157 [-0.001163, +0.000843] | -0.045242 [-0.049080, -0.041597] |
| Bicarbonate | +0.001855 [-0.002143, +0.005844] | -0.032220 [-0.043201, -0.021802] |

Creatinine improves pooled MAE versus the prior selector, but that gain does
not pass every subgroup check and it still significantly loses to persistence.
BUN's small improvement versus the selector is not statistically established.
Urine significantly worsens versus the selector and always-expert. Potassium's
incremental gain is tiny and uncertain. Bicarbonate does not improve the selector.

Evidence-only adoption beats the full-state primary in pooled creatinine,
urine and potassium comparisons. BUN does not beat the shuffled-input control
(3.058475 versus 3.058354). Adding patient-state/context inputs is therefore
not demonstrated to be the source of a stable new gain.

## What the Amplitude Experiment Diagnoses

The eICU router-fit optimal global alpha is **1.0 for all five targets**.
The fixed expert also has lower eICU patient-test MAE than conditional adoption
on all five. A universally smaller gain is not supported by source training.

Mean conditional alpha in MIMIC is 0.758, 0.857, 0.735, 0.863 and 0.750,
respectively. Lowering amplitude helps stable cases but compromises dynamic
cases relative to the binary selector:

| Target | Stable MAE: binary -> amplitude | Dynamic MAE: binary -> amplitude |
|---|---|---|
| Creatinine | 0.109462 -> 0.106562 | 0.655614 -> 0.661780 |
| BUN | 1.770694 -> 1.746665 | 8.675125 -> 8.739875 |
| Urine output | 24.550351 -> 22.668799 | 157.213037 -> 159.245521 |
| Potassium | 0.199886 -> 0.184912 | 0.572206 -> 0.587225 |
| Bicarbonate | 0.795180 -> 0.766118 | 2.992211 -> 3.025893 |

Stable/dynamic is defined retrospectively by actual absolute change thresholds
(0.3, 5, 50, 0.3, 2). It is never an input, is not equivalent to healthy/sick,
and does not provide an oracle label available to a deployed selector.

The repair addresses binary all-or-nothing adoption and partially reduces
overshoot. It does NOT solve identification of the appropriate change magnitude
under transfer. These results are consistent with conditional distribution
shift, limited informative features, finite-sample fitting error, and remaining
measurement/treatment harmonization gaps. They do not isolate one of those as
the sole cause, or prove that no other architecture or data can help.

## Uncertainty Results

All five primary arms pass coverage in eICU patient and hospital evaluation.
None pass MIMIC coverage under the unchanged 0.87-0.93 policy:

| Target | Patient-equal coverage | Unit-equal coverage | Mean width | CRPS vs persistence distribution |
|---|---:|---:|---:|---|
| Creatinine | 0.96900 | 0.95833 | 1.14820 | Improved |
| BUN | 0.95592 | 0.96927 | 17.32540 | Worse |
| Urine output | 0.94725 | 0.93598 | 581.70772 | Improved |
| Potassium | 0.93875 | 0.91017 | 1.89548 | Improved |
| Bicarbonate | 0.96142 | 0.94741 | 11.52485 | Worse |

These failures are overcoverage, not undercoverage. They suggest inefficient
or mismatched source-calibrated uncertainty, not proof of dangerous optimism.
Marginal overcoverage does not exclude undercoverage in a particular subgroup.
Per-unit descriptive counts/coverage are in the report. Do not tighten widths
using the same MIMIC outcomes and call the result external validation.

## Limitations and What Remains

- This family can only shrink a frozen expert delta. It cannot reverse an
  incorrect direction, amplify an underestimated change, or recover an event
  absent from the available pre-anchor information.
- Seven patient resampling seeds are not seven trained checkpoints. The
  late-relative-time subset is not a calendar-time split; unit-equal inference
  is not evidence that every care unit benefits.
- Repeated experiments on the same transfer cohort cannot establish a final
  generalization claim. No newly independent lockbox was used here.
- Precise future-event provenance remains asymmetric between these transition
  files. This run did not alter labels or establish label mismatch as the cause.
- If site-specific adaptation is pursued, it needs separate local calibration
  patients and a subsequent untouched local evaluation. That answers adaptation,
  not the original zero-shot eICU-to-MIMIC question.

## Artifacts and Verification

- `artifacts/amplitude_calibration_20260905/development.json`
- `artifacts/amplitude_calibration_20260905/report.json`
- `artifacts/amplitude_calibration_20260905/amplitude_heads.joblib`
- `whole_body_jepa/amplitude_calibration.py`
- `scripts/audit_amplitude_calibration.py`
- `tests/test_amplitude_calibration.py`

The joblib contains readouts and per-arm interval scales, not a standalone
replacement for the frozen JEPA and expert bundle. Load only trusted local
joblib files. No patient-row predictions or identifiers are exported.

Reproduce in a fresh output directory with the local environment and existing
raw-data defaults:

```sh
env OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 LOKY_MAX_CPU_COUNT=6 \
  /Users/chunyouchang/Desktop/llm_project/medical_jepa/.venv/bin/python \
  -m scripts.audit_amplitude_calibration \
  --output-dir artifacts/amplitude_calibration_reproduction
```

192 regression tests pass, including 7 new amplitude tests. Release validation
passes with 41 catalog / 27 move / 14 fallback. No release replacement, no
causal claims, no clinical decision claims, and no threshold relaxation.
