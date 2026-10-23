# Measurement Harmonization Results (2026-10-15)

Decisions approved 2026-10-15 (`MEASUREMENT_DEFINITION_INVENTORY_20261015.md`) and how they were
implemented:

| Item | Implementation |
|---|---|
| urine output | eICU modules rebuilt with the current interval-normalized mL/hour derivation (eICU median 51, p90 200; MIMIC 60, 200; old eICU files 110, 585) |
| glucose | eICU `bedside glucose` (laboratory table and nurse charting) → `glucose_poc`; the `glucose` target is laboratory or blood gas only in all databases. MIMIC adds `glucose_poc` from chartevents 225664 (fingerstick). NWICU has no point-of-care item. |
| vital signs | eICU heart rate, respiratory rate, SpO2, temperature and MAP now come from nurse-validated `nurseCharting` (measured means only for MAP) instead of automatic `vitalPeriodic`. eICU heart-rate age: median 0.4h (was 0.0h); MIMIC 0.5h. |
| labels | eICU uses the extractor label rule shared with MIMIC and NWICU (nearest measurement within min(2, h/2) hours); the separate `eicu_raw_measurement_time_augment.py` relabeling is not applied. |
| MAP (NWICU) | unchanged; declared derived-MAP deviation |
| electrolytes, hemoglobin | blood-gas share in MIMIC labevents (hospital-wide): potassium 7.4%, sodium 3.7%, hemoglobin 3.3%, hematocrit 3.2%, bicarbonate 0.8%; NWICU potassium 7.8%, sodium 7%. Judged not material, not harmonized. The ICU-only share is likely somewhat higher. |

Code: `eicu_sepsis_transition_extract.py` (canonical measurement cache, `read_nurse_vitals`, POC
mapping; `read_vitals` from vitalPeriodic removed), `build_eicu_measurement_cache.py`,
`eicu_body_system_transition_extract.py` (`input_state_vars`), AKI and respiratory state variables,
`mimiciv_observation_transition_extract.py` (`glucose_poc`). Tests:
`test_eicu_measurement_harmonization.py`.

Data (v3): `mimiciv_observation_transitions_{h}h_full_v31_treatment_unit_safe_v3.parquet`,
`nwicu_observation_transitions_{h}h_v3.parquet`, `eicu_harmonized_20261015/` (8 modules used by the
audits: aki, sepsis, hepatic_failure, coagulopathy_heme at 1–48h; respiratory, endocrine_stress,
cardiovascular_instability, electrolyte_acid_base at 1–12h). The other 7 eICU modules were not
rebuilt and still use the old definitions.

## Effect on the v2-interval audits (non-independent)

| | Before (v2 data) | After (v3 data) |
|---|---|---|
| eICU point pass (49) | 33 | 34 |
| eICU full pass, v2 interval | 16 | **18** |
| NWICU full pass, v2 interval (40 evaluable) | 30 | 29 |

eICU changes by target (patient-equal coverage, v2 interval):

- **urine_output:** 1h 0.793 → 0.863, 3h 0.789 → **0.906**, 6h 0.687 → **0.918**. The unit fix
  resolves most of the urine failure; urine@3h and urine@6h now pass fully.
- **o2sat@12h:** 0.743 → 0.841 (nurse-charted SpO2).
- **respiratory_rate:** 6h 0.879 → 0.893, 12h 0.866 → 0.877 (pass), 24h 0.867 → 0.890 (pass).
- **heart_rate:** about 0.01 lower in every cell; heart_rate@6h drops out (0.887 → 0.875).
- **glucose:** short horizons get worse (1h 0.856 → 0.769, 6h 0.802 → 0.760); long horizons
  improve (24h 0.871 → 0.887, 48h 0.851 → 0.874). Laboratory-only eICU glucose pairs are fewer and
  come from a different subset of patients. The new `glucose_poc` input may also differ in
  meaning between MIMIC fingerstick and eICU bedside records. Not yet explained.
- Other losses: potassium@6h, bicarbonate@24h.

NWICU barely moves (every coverage change within ±0.01), as expected: only the MIMIC side gained
`glucose_poc`. Refitting on v3 data loses creatinine@6h and glucose@24h and gains creatinine@12h.

## Relation to the promotion registry

`external_expert_promotion_registry_20261016.json` (superseding the 2026-10-15 version) is pinned to the v2 MIMIC data. Recipes refit on
v3 data are not covered by it. Moving promoted cells to v3 recipes requires new validation; on
current non-independent evidence glucose@24h (tier A) and creatinine@6h (tier B) would not hold
on NWICU.

`causal_claim_allowed: false`. Nothing promoted by this step.
