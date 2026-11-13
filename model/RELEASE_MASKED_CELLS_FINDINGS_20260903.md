# Frozen Release Masked-Cell Audit

## Result

Catalog: 41 cells, corresponding to 15 current numeric variables plus an interval-event target.
MIMIC: 15 current targets and 38 numeric forecast cells evaluated.
NWICU: 14 current targets and 37 numeric forecast cells evaluated.
Full current reconstruction diagnostic passes: MIMIC `[]`; NWICU `[]`.
Seven patient-split passes before the care-unit gate: MIMIC `['hematocrit', 'platelets']`; NWICU `['hematocrit', 'hemoglobin', 'platelets']`.
Masked forecast source passes: MIMIC `[['creatinine@12h', 'frozen_whole_body_jepa']]`; NWICU `[]`.
The two event cells require a separate event label contract. Urine has no NWICU label;
platelets@1h has no eligible unused MIMIC patients in the frozen head exclusion pool.
Unsupported cells are not model failures. No candidate is promoted by this audit.

## Protocol

All weights, source choices and scalers are frozen; no runtime changes or promotions.
Mask the target and predefined siblings from every input/history row. Recompute
context, belief, base point, organ and target features. Missing anchors use the
checkpoint population prior, not the hidden value. Serving's neutral workflow
mask/age contract is preserved. Labels, provenance and future treatments are not inputs.
One hash-selected eligible anchor per patient, at most 4000 patients per task; not a full-row census.
Same-anchor truth must be measured within 1 hour. Future labels are separate.
Compare with the frozen population prior and three wrong-patient context controls.
The latter are same-frozen-model derangements, NOT separately trained shuffled-label models.
Require seven overlapping patient split CIs and every supported careunit CI below zero.
Units need 30 patients; at least two units. Splits need 100 test patients.
These are nominal diagnostic screens, not multiplicity-corrected promotion gates.
Existing forecast intervals are invalid under masking and are not reused.
Both databases have been inspected previously; neither is an untouched lockbox.

## Same-anchor reconstruction

This tests the packaged checkpoint's reconstruction head, NOT old retrained candidates.
A target's current reconstruction result is reused across horizons, not counted repeatedly.

| Target | MIMIC MAE / population MAE | MIMIC diagnostic | NWICU MAE / population MAE | NWICU diagnostic |
|---|---|---|---|---|
| bicarbonate | 4.151 / 4.083 | False | 3.867 / 3.867 | False |
| bun | 14.86 / 14.25 | False | 14.34 / 13.95 | False |
| chloride | 7.027 / 4.96 | False | 7.649 / 4.863 | False |
| creatinine | 0.8397 / 0.7978 | False | 0.8265 / 0.7747 | False |
| glucose | 37.88 / 35.95 | False | 39.99 / 39.26 | False |
| heart_rate | 14.24 / 13.88 | False | 14.32 / 14.1 | False |
| hematocrit | 4.931 / 5.183 | False | 5.439 / 5.737 | False |
| hemoglobin | 1.694 / 1.706 | False | 1.878 / 1.959 | False |
| map | 12.83 / 13.25 | False | 11.81 / 11.97 | False |
| o2sat | 2.246 / 2.209 | False | 2.409 / 2.382 | False |
| platelets | 73.3 / 76.21 | False | 74.78 / 76.42 | False |
| potassium | 0.4636 / 0.4636 | False | 0.4375 / 0.4318 | False |
| respiratory_rate | 4.471 / 4.316 | False | 4.688 / 4.59 | False |
| sodium | 4.818 / 4.082 | False | 4.567 / 3.851 | False |
| urine_output | 64.36 / 64.93 | False | insufficient_fresh_observations | False |

MAE is in the target's physical units, not a percentage accuracy.

## All catalog cells

Forecast heads do not define a same-anchor nowcast. Their forced missing-anchor
predictions below are stress tests, not new serving capabilities. Persistence
cannot be computed without its anchor; unmasked persistence is diagnostic only.

| Cell | Runtime source | Current reconstruction (MIMIC / NWICU) | Masked future sources passing MIMIC | Passing NWICU |
|---|---|---|---|---|
| glucose@1h | frozen_whole_body_jepa | False / False | none | none |
| hypoxemia_event@1h | frozen_whole_body_jepa | False / False | requires_binary_interval_event_labels | requires_binary_interval_event_labels |
| map@1h | frozen_whole_body_jepa | False / False | none | none |
| platelets@1h | patient_persistence_anchor | False / False | insufficient_supported_patients | none |
| urine_output@1h | unified_shared_jepa_target_head | False / False | none | no_compatible_future_target |
| glucose@3h | unified_shared_jepa_target_head | False / False | none | none |
| heart_rate@3h | patient_persistence_anchor | False / False | none | none |
| hematocrit@3h | patient_persistence_anchor | False / False | none | none |
| hemoglobin@3h | patient_persistence_anchor | False / False | none | none |
| hypoxemia_event@3h | frozen_whole_body_jepa | False / False | requires_binary_interval_event_labels | requires_binary_interval_event_labels |
| map@3h | frozen_whole_body_jepa | False / False | none | none |
| platelets@3h | patient_persistence_anchor | False / False | none | none |
| respiratory_rate@3h | patient_persistence_anchor | False / False | none | none |
| urine_output@3h | unified_shared_jepa_target_head | False / False | none | no_compatible_future_target |
| creatinine@6h | patient_persistence_anchor | False / False | none | none |
| glucose@6h | unified_shared_jepa_target_head | False / False | none | none |
| heart_rate@6h | multisource_shared_jepa_target_head | False / False | none | none |
| hematocrit@6h | patient_persistence_anchor | False / False | none | none |
| hemoglobin@6h | patient_persistence_anchor | False / False | none | none |
| map@6h | frozen_whole_body_jepa | False / False | none | none |
| respiratory_rate@6h | multisource_shared_jepa_target_head | False / False | none | none |
| bicarbonate@12h | patient_persistence_anchor | False / False | none | none |
| chloride@12h | unified_shared_jepa_target_head | False / False | none | none |
| creatinine@12h | patient_persistence_anchor | False / False | frozen_whole_body_jepa | none |
| glucose@12h | unified_shared_jepa_target_head | False / False | none | none |
| heart_rate@12h | frozen_whole_body_jepa | False / False | none | none |
| hematocrit@12h | unified_shared_jepa_target_head | False / False | none | none |
| hemoglobin@12h | unified_shared_jepa_target_head | False / False | none | none |
| map@12h | frozen_whole_body_jepa | False / False | none | none |
| o2sat@12h | frozen_whole_body_jepa | False / False | none | none |
| potassium@12h | frozen_whole_body_jepa | False / False | none | none |
| respiratory_rate@12h | frozen_whole_body_jepa | False / False | none | none |
| bicarbonate@24h | multisource_shared_jepa_target_head | False / False | none | none |
| bun@24h | patient_persistence_anchor | False / False | none | none |
| heart_rate@24h | unified_shared_jepa_target_head | False / False | none | none |
| map@24h | unified_shared_jepa_target_head | False / False | none | none |
| potassium@24h | frozen_whole_body_jepa | False / False | none | none |
| respiratory_rate@24h | frozen_whole_body_jepa | False / False | none | none |
| sodium@24h | unified_shared_jepa_target_head | False / False | none | none |
| glucose@48h | patient_persistence_anchor | False / False | none | none |
| heart_rate@48h | multisource_shared_jepa_target_head | False / False | none | none |

## Interpretation limits

Failure can reflect missing patient-specific information or masked-input distribution
shift; it does not prove a variable is fundamentally unpredictable. Success supports
conditional recoverability, not complete physiology, causality or hidden-state truth.
The shipped model was not block-mask trained identically to the newer reconstruction
candidates. Their historic passes cannot be attributed to this checkpoint.
Urine lacks a matching NWICU output contract. Binary hypoxemia requires its own
interval-event labels; current O2sat reconstruction is not event-risk validation.

## What failed

The frozen checkpoint was trained for forecasts anchored to an observed target.
Its reconstruction branch did not receive the newer, matched full-history/sibling
block-mask pretraining. This audit tests that packaged branch, not those later
reconstruction candidates. Contract mismatch is a plausible contributor, not a
proven sole cause. Low population MAE differences alone are not reliable recovery.
Hematocrit and platelets show patient-level recovery evidence in both databases,
but not significant improvements in every supported care unit. A failed unit CI
may cross zero; it does not necessarily mean the point estimate regressed.
The all-supported-unit diagnostic gate here is stricter than the pooled/equal-unit
forecast serving gate. Counts are therefore not directly comparable to the release's
28 movable cells. Removing the entire target history is also a different information
setting from normal observed-anchor forecasting and may be out of training support.
Thus this result does not revoke normal forecast validation, and does not prove
the latent contains no signal. It rejects universal masked-input recovery for this release.

## Reproduce

```bash
python -m scripts.audit_release_masked_cells
```

Override the documented CLI paths for local canonical examples and unused-patient
cache. Model and input hashes are recorded. Only aggregate results are written.
