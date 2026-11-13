# Measurement Definition Inventory: MIMIC-IV, eICU, NWICU (2026-10-15)

Sources: extractor item maps (`mimiciv_observation_transition_extract.py`,
`eicu_sepsis_transition_extract.py`, `nwicu_observation_transition_extract.py`),
`EICU_RESIDUAL_GAP_CAUSES_FINDINGS_20261015.md`, file dates.

## Confirmed mismatches

| Target | MIMIC-IV (v2 rebuild) | eICU module files | NWICU | Impact |
|---|---|---|---|---|
| **urine_output** | interval-normalized **mL/hour** (median 60, p90 200) | **raw charted volume**: files built 2026-06-27 / 07-15, before the 2026-08-02 normalization (median 100–125, p90 500–625) | not recorded | **All eICU urine results from 2026-10-14 onward compare different units and are invalid.** The frozen 2026-08-16 audit used volume in both databases. |
| **glucose** | laboratory + blood gas only | laboratory **+ bedside point-of-care** (72–88% of labels) | laboratory + blood gas | bedside labels cover 0.02–0.04 less |
| **map** | device-measured mean (arterial 220052, non-invasive 220181) | arterial mean (vitalPeriodic, 5-minute) + non-invasive mean (vitalAperiodic) | **derived** (SBP + 2·DBP)/3 | declared deviation for NWICU |
| **heart_rate, respiratory_rate, o2sat** | nurse-charted chartevents (about hourly, validated) | **vitalPeriodic automatic 5-minute values** (not nurse-validated) | nurse-charted | different sampling and validation; o2sat@12h is the worst eICU cell |
| **potassium, sodium** | chemistry + blood-gas whole blood | laboratory names `potassium` / `sodium` | chemistry + blood-gas whole blood | small; not yet quantified |
| **hemoglobin, hematocrit** | CBC + blood gas | CBC (`Hgb`, `Hct`) | CBC | small; not yet quantified |
| **bicarbonate** | chemistry + calculated blood gas | `bicarbonate` + `HCO3` | Total CO2 | small |
| creatinine, BUN, platelets, chloride | consistent | consistent | consistent | none found |

## Proposed canonical definitions (need a decision before re-extraction)

1. **urine_output:** interval-normalized mL/hour everywhere. Re-extract the eICU module files with
   the current extractors (`derive_interval_urine_rate`).
2. **glucose:** target = laboratory or blood-gas glucose only. Point-of-care fingerstick becomes a
   separate input `glucose_poc` (MIMIC chartevents fingerstick items; eICU `bedside glucose`).
   Trade-off: eICU glucose labels drop by about 72–88%.
3. **Vital signs:** take eICU vitals from nurse-validated charting (`nurseCharting`), closer to
   MIMIC chartevents, instead of automatic `vitalPeriodic` values; keep vitalPeriodic as an
   audit comparison.
4. **MAP:** keep measured mean; NWICU stays a declared derived-MAP deviation.
5. **Electrolytes and hemoglobin:** quantify the blood-gas share first; harmonize only if material.

Re-extraction cost: eICU modules (15 modules × 6 horizons, raw vitalPeriodic and nurseCharting
scans) and a MIMIC rebuild for `glucose_poc`. Several hours of compute. All downstream eICU
results must then be re-run.
