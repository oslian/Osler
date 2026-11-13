# Full-Coverage Shared-Body Reconstruction

Report: `artifacts/native_reconstruction_full_coverage_20260913.json`.
All recorded dependency hashes verified. Same 35 targets, 32 fit patients and
16 patients in each tune/evaluation/hospital role; 879/442/422/427 respective
patient-target examples. No cohort or labels rebuilt using the new outcomes.

Each arm received 1,120 updates per epoch for eight epochs (8,960 total).
Balanced scheduling asserts every fit pair is consumed each epoch; repeating
rare targets increases optimization exposure, not independent patient evidence.

| Arm | Selected epoch | Target-equal normalized tune MAE |
|---|---:|---:|
| Initial fit-median readout | 0 | 1.811102 |
| Event-weighted | 0 | 1.811102 |
| Full-coverage balanced | 7 | 1.790593 |

Event-weighted final tune MAE is 1.830667: every evaluated trained epoch is
worse than initialization. Balanced final tune MAE is 1.799081. Original
checkpoint selection including epoch zero is retained.

## Descriptive Heldout Results

| Role | Reported targets | Population MAE | Balanced MAE | Numerically better targets |
|---|---:|---:|---:|---:|
| Evaluation | 34 | 1.761483 | 1.739840 | 21 |
| Hospital | 32 | 1.046497 | 1.029325 | 17 |

These are target-equal normalized errors, not physical-unit errors. Target
sets differ across roles; do not interpret their difference as transfer gain.
Event-weighted selected output is exactly population (epoch zero).

Eleven targets improve numerically in both roles: albumin, AST, glucose,
hematocrit, hemoglobin, MAP, platelets, respiratory rate, sodium, temperature,
and WBC. This exploratory list is not an eligibility or promotion list.

**Every comparison has `supported: false` and `ci95: null`.** The paired
bootstrap in `whole_body_jepa/change_evaluation.py` requires 30 unique patients;
each role here has at most 16. This pilot cannot pass any statistical gate
regardless of accuracy. Earlier small sampling/probe results also indicate
insufficient evidence, not statistically established absence of information.

## Diagnosis And Decision

The previous sampler underexposed patient-target pairs. Full coverage changes
learning and gives a small descriptive heldout improvement, but does not yet
establish reliable gain. Schedule and budget both changed from the old pilot;
their independent contributions are not isolated.

The validation sample is structurally below the reporting threshold. Do not
lower that threshold. Scale the original-event inventory with source-defined
inclusion and preflight per-target patient support before further training.
Thirty patients is a minimum reporting condition, not adequate-power proof.
Do not select only these eleven exploratory winners.

This is current masked reconstruction, not future forecasting. Capacity-matched
placebo training, seven-seed confirmation, calibration and new external data
remain needed. Reused holdouts are not untouched. Eleven input/reconstruction
tests passed; no release weights, registry, intervals or runtime changed.
