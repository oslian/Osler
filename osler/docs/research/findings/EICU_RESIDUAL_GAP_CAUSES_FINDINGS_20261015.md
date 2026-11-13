# Causes of the Remaining eICU Coverage Gap (2026-10-15)

With v2 value-conditional intervals, eICU patient-equal coverage stays below 0.90 at the same
current value (median gap −0.040 across 47 continuous cells). Diagnostics:
`eicu_within_bin_gap_diagnostic.py`, `eicu_glucose_source_diagnostic.py`,
`eicu_module_cohort_diagnostic.py`. eICU labels were used for analysis only.

## Ruled out or minor

| Hypothesis | Test | Result |
|---|---|---|
| H1 label timing (eICU keeps labels 2–3.6h off target) | eICU exact-time labels only | median Δ 0.000; never helps more than +0.014 |
| H2 missing inputs (eICU modules: 13–27 state / 10–40 treatment columns vs 81 / 279) | MIMIC-only masking of eICU-absent inputs | median Δ −0.003, worst −0.016 |
| H5 any in-window treatment | eICU coverage with vs without | small, inconsistent; 62–99% of rows have some treatment |

## Main finding: heavier tails of real change, driven by cohort definition

- The gap tracks the **tail** of real target change at the same current value. Spearman of
  the gap against the eICU/MIMIC ratio of the 90th-percentile persistence error |future − current|
  is −0.58 (p = 2e-5). Against the model-residual p90 ratio it is −0.74 (p = 3e-9). Median ratios
  are about 1: typical patients look alike, extreme changes do not.
- **Cohort definition drives the tail.** eICU transitions are built per disease module. Scoring
  the same MIMIC recipe on the same target in different modules:

| Cell | Most volatile module | Least volatile module |
|---|---|---|
| o2sat@12h | respiratory 0.743 | acute_neuro 0.907 |
| glucose@6h | endocrine_stress 0.802 | acute_neuro 0.902 |
| creatinine@6h | aki 0.759 | acute_neuro 0.851 |
| potassium@24h | hepatic_failure 0.864 | coagulopathy_heme 0.917 |
| map@6h | respiratory 0.852 | coagulopathy_heme 0.883 |

  The external audit's fixed module order (`PREFERRED_MODULE`) scores each target in its own
  disease cohort (creatinine→AKI, glucose→endocrine stress, o2sat→respiratory), which is the
  most volatile population for that target. The eICU test therefore measures transfer to the
  disease-enriched subgroup, not to a general ICU population. NWICU, a whole-ICU cohort like
  MIMIC, passes 30/40 under v2.

## Measurement method: glucose

- MIMIC glucose uses laboratory and blood-gas items only. The eICU extraction merges
  `bedside glucose` (point-of-care fingerstick) with laboratory glucose.
- 72–88% of eICU glucose labels are bedside (100% of rows matched to raw `lab.csv`).
- Bedside-labelled rows cover 0.02–0.04 less than laboratory-labelled rows at every horizon
  (6h: 0.793 vs 0.831; 24h: 0.850 vs 0.887). Laboratory→laboratory rows at 24h and 48h reach
  0.888 and 0.881.
- The fingerstick mismatch explains part of the glucose gap; the endocrine-stress cohort
  explains more.

## Unexplained remainder

Some cells stay below 0.87 even in their least volatile module (creatinine@6h best 0.851).
Candidates not tested here: severity or diagnosis beyond module membership, practice era
(eICU 2014–2015), and other assay or definition differences (urine output definitions are
known to differ).

## Implications

1. The eICU gap is mostly **population definition**: the model is less calibrated for patients
   who have the disease the target is about. This is clinically relevant, not an artifact to tune
   away.
2. A principled fix is to condition intervals on disease state: rebuild the eICU module
   definitions in MIMIC and calibrate per module (Mondrian by module × value), MIMIC only.
3. Glucose needs measurement harmonization: model point-of-care and laboratory glucose as
   separate sources (MIMIC fingersticks are in chartevents), or score eICU laboratory glucose
   separately.

`causal_claim_allowed: false`. Diagnostic only; nothing promoted.
