# Disease-Module-Conditional Intervals (v3): Findings (2026-10-15)

Scripts: `icu_disease_module_membership.py` (membership for MIMIC-IV and NWICU),
`module_conditional_interval_audit.py` (v3). Detail tables:
`MODULE_CONDITIONAL_INTERVAL_eICU_FINDINGS_20261015.md`,
`MODULE_CONDITIONAL_INTERVAL_NWICU_FINDINGS_20261015.md`. Every v2 number reproduced the v2 audit
exactly before v3 was applied.

## Method

v3 = Mondrian by membership in the target's own disease module (creatinine/BUN/urine → AKI,
glucose → endocrine stress, MAP/heart rate → cardiovascular instability, o2sat/respiratory rate →
respiratory, potassium/sodium/chloride/bicarbonate → electrolyte/acid-base, hemoglobin/hematocrit/
platelets → coagulopathy/heme) × v2 value normalization. Both groups are calibrated on MIMIC only.
The member pool adds up to 15,000 rows from MIMIC member patients never used for fitting.

## 1. The disease-subgroup effect is real inside MIMIC

On held-out MIMIC module members, the whole-MIMIC v2 interval under-covers: median 0.889; 10/46
cells below 0.87; creatinine 0.82–0.86, glucose@6h 0.833, o2sat@12h 0.845. The eICU shortfall is
therefore a genuine subgroup effect, not an eICU-specific artifact. v3 restores MIMIC member
coverage to 0.899–0.902 (5-fold CV).

## 2. External effect: helps members, hurts non-members

| | Members v2 → v3 | Non-members v2 → v3 | Full pass v2 → v3 |
|---|---|---|---|
| eICU (non-independent) | 0.856 → 0.873 | 0.879 → 0.865 | 14 → 15 of 47 continuous |
| NWICU (non-independent) | 0.876 → 0.888 | 0.910 → 0.897 | 28 → 24 of 38 continuous |

- eICU disease cohorts improve where membership is meaningful: creatinine@6h 0.759 → 0.821,
  @12h 0.779 → 0.861, @24h → pass, @48h → pass; bun@24h → pass; glucose@6h 0.802 → 0.876;
  o2sat@12h 0.743 → 0.800; respiratory_rate@12h and @24h → pass.
- eICU cells drawn from a fallback module (2–4% target-module members, for example glucose@24h
  from the AKI cohort) get worse. Their patients are sick in another way, but receive MIMIC's
  narrower non-member interval.
- NWICU, a general ICU population, gets worse overall. Its AKI membership (54–61% of creatinine
  rows, mostly from the KDIGO rule) does not mean the same thing as MIMIC's (creatinine@6h
  care-unit coverage 0.918 → 0.796).

## 3. Why v3 is not adopted

1. **Membership is not portable.** MIMIC's `diagnoses_icd` lists every discharge diagnosis,
   comorbidities included (AKI 41%, coagulopathy/heme 61% of stays). NWICU coding is sparser
   (14–25%), and eICU modules use problem-list and admission diagnoses. The same label marks
   different populations in each database.
2. **Membership uses information not available at prediction time.** ICD codes are assigned at
   discharge, and the KDIGO and SpO2 rules scan the whole stay, including the future. v3's gains
   could not be reproduced at runtime.

The runtime registry is unchanged. v2 remains the value-conditional method.

## Recommended next step (v4)

Condition on **as-of, measurement-defined disease activity** instead of diagnosis codes, computed
identically in every database from data available before the anchor. Examples: a creatinine rise
over the prior 48h, glucose above 250 or an active insulin infusion, SpO2 ≤ 90 or ventilation,
platelets below 100, MAP below 65 or vasopressors. The eICU module configs already define such
`active_low` / `active_high` thresholds. This is portable, leak-free and deployable. It would be
selected on MIMIC only and still needs an untouched database for independent validation.

`causal_claim_allowed: false`. Non-independent re-check; nothing promoted.
