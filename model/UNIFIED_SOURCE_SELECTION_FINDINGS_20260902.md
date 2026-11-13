# Unified Frozen JEPA Source Selection

## Result

- Catalog: 41 cells; compared numeric cells: 37.
- New source replacements: 4.
- Selected source with both new point/interval gates: 5.
- Unified runtime movable cells: 28 (overlaps, not additive).
- Runtime sources: `{'frozen_whole_body_jepa': 11, 'calibrated_event_risk_head': 2, 'patient_persistence_anchor': 13, 'unified_shared_jepa_target_head': 11, 'multisource_shared_jepa_target_head': 4}`.

## Protocol

All three readouts use one frozen checkpoint and the same 311-column representation.
Base JEPA uses its value output. Original shared heads retain their exact frozen
runtime normalization; multi-source heads use their fitted development normalization.
There is no new weight fitting, target-domain normalization, or test-time refitting.

For every cell, reconstruct and exclude all MIMIC patients in the multi-source
head's original selection pool, not just individual fitted rows. Split remaining
patients by a fixed hash into source selection (40%), calibration (30%) and
final development test (30%). Source selection requires seven fixed patient
split comparisons, significant improvement over the incumbent and persistence,
then ranks eligible challengers by patient-equal MAE. These overlapping splits
are robustness checks, not seven independent replications.

Freeze the chosen source before reading NWICU outcomes. Calibration uses one
outcome-independent anchor per calibration patient and the exact 90% order statistic.
MIMIC test and NWICU must pass patient/careunit/late-time paired bootstrap versus
persistence and the incumbent, plus row/patient/unit/late coverage in [0.87,0.93].
Direction accuracy and delta correlation are reported as diagnostics.
Failure does not trigger trying the runner-up on the external test: retain the
incumbent with its own original interval and original validation scope.

Both databases have previously been inspected during research. This is a reproducible
development/transfer audit, NOT a new untouched lockbox or prospective clinical test.
Careunit gates aggregate supported units; no guarantee for every small subgroup.
Released timestamps are not original calendar time. No causal claim is made.

## Per-cell decisions

| Cell | Incumbent | Development choice | Runtime source | Changed | New joint gate | Failure |
|---|---|---|---|---|---|---|
| glucose@1h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | mimic_test:point, mimic_test:careunit_coverage, nwicu:point, nwicu:careunit_coverage |
| hypoxemia_event@1h | frozen_whole_body_jepa | frozen_whole_body_jepa | frozen_whole_body_jepa | False | False | no_matching_external_numeric_contract |
| map@1h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | nwicu:point |
| platelets@1h | patient_persistence_anchor | patient_persistence_anchor | patient_persistence_anchor | False | False | insufficient_unused_development_patients |
| urine_output@1h | unified_shared_jepa_target_head | unified_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | no_matching_external_numeric_contract |
| glucose@3h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | mimic_test:point, nwicu:point |
| heart_rate@3h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | nwicu:point |
| hematocrit@3h | patient_persistence_anchor | unified_shared_jepa_target_head | patient_persistence_anchor | False | False | nwicu:point |
| hemoglobin@3h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | nwicu:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage |
| hypoxemia_event@3h | frozen_whole_body_jepa | frozen_whole_body_jepa | frozen_whole_body_jepa | False | False | no_matching_external_numeric_contract |
| map@3h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | nwicu:point |
| platelets@3h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | mimic_test:point, nwicu:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage |
| respiratory_rate@3h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | mimic_test:point |
| urine_output@3h | unified_shared_jepa_target_head | unified_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | no_matching_external_numeric_contract |
| creatinine@6h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | mimic_test:late_coverage |
| glucose@6h | unified_shared_jepa_target_head | unified_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | mimic_test:point, mimic_test:late_coverage |
| heart_rate@6h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | multisource_shared_jepa_target_head | True | True | - |
| hematocrit@6h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | nwicu:point |
| hemoglobin@6h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | nwicu:point, nwicu:coverage, nwicu:careunit_coverage |
| map@6h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | mimic_test:late_coverage, nwicu:point, nwicu:late_coverage |
| respiratory_rate@6h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | multisource_shared_jepa_target_head | True | True | - |
| bicarbonate@12h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | nwicu:point |
| chloride@12h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | mimic_test:point, nwicu:point, nwicu:coverage, nwicu:careunit_coverage |
| creatinine@12h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | mimic_test:coverage, mimic_test:careunit_coverage, mimic_test:late_coverage |
| glucose@12h | unified_shared_jepa_target_head | unified_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | mimic_test:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage |
| heart_rate@12h | frozen_whole_body_jepa | frozen_whole_body_jepa | frozen_whole_body_jepa | False | False | mimic_test:late_coverage |
| hematocrit@12h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | nwicu:point, nwicu:coverage, nwicu:careunit_coverage |
| hemoglobin@12h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | nwicu:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage |
| map@12h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | nwicu:point |
| o2sat@12h | frozen_whole_body_jepa | frozen_whole_body_jepa | frozen_whole_body_jepa | False | True | - |
| potassium@12h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | mimic_test:point, nwicu:point |
| respiratory_rate@12h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | mimic_test:point, nwicu:point |
| bicarbonate@24h | patient_persistence_anchor | multisource_shared_jepa_target_head | multisource_shared_jepa_target_head | True | True | - |
| bun@24h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | mimic_test:late_coverage |
| heart_rate@24h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | mimic_test:point |
| map@24h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | nwicu:point |
| potassium@24h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | mimic_test:point, mimic_test:late_coverage |
| respiratory_rate@24h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | frozen_whole_body_jepa | False | False | nwicu:point |
| sodium@24h | unified_shared_jepa_target_head | multisource_shared_jepa_target_head | unified_shared_jepa_target_head | False | False | mimic_test:coverage, mimic_test:careunit_coverage, mimic_test:late_coverage, nwicu:point |
| glucose@48h | patient_persistence_anchor | multisource_shared_jepa_target_head | patient_persistence_anchor | False | False | mimic_test:point, nwicu:coverage, nwicu:careunit_coverage, nwicu:late_coverage |
| heart_rate@48h | frozen_whole_body_jepa | multisource_shared_jepa_target_head | multisource_shared_jepa_target_head | True | True | - |

## Serving

`whole_body_jepa.runtime.predict_patient` loads the hash-bound selector and
runs the checkpoint once per horizon. Each target reads only its selected head.
The interval is inseparable from that source: new source/new calibration or
retained source/original calibration. Never use the old interval for a new point.
Four urine/event cells lack a matching NWICU target contract; they retain their
existing models and are explicitly not counted as newly validated.
Missing current values remain missing; input zero remains observed. A missing
selector or mismatched weights stops loading instead of silently changing models.
No next-measurement-time prediction, new clinical indication, or latent-coupling
proof is introduced. All patient-level caches stay outside this repository.

## Interval-only rescue

The four fallback cells with both point gates already passed were tested
with one fixed conditional scale model. It uses current value and absolute
predicted change, fits on selection patients, and calibrates on separate
patients at nominal 90%. No external fitting or point-weight changes.
Promoted cells: `['bicarbonate@24h']`.
Other cells retain their prior source and interval. This follow-up reuses
research validation data and is not independent confirmatory evidence.
Details: [interval rescue](PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md).
