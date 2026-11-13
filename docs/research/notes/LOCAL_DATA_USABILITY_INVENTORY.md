# Local Data Usability Inventory

Audit date: 2026-07-11. This inventory separates data that can directly feed
the current observation/prediction contracts from data that still needs an
extractor, a target definition, or an external validation gate.

## Executive Summary

The machine contains four valuable real-world data families:

1. full MIMIC-IV v3.1;
2. full eICU Collaborative Research Database 2.0;
3. MIMIC-IV-ED and MIMIC-IV clinical notes;
4. MIMIC-IV Waveform Database.

The most important underused resource is not another ICU vital sign. It is
the MIMIC-IV hormone laboratory data, especially TSH, free T4, total T4,
cortisol, and PTH. These are real observations and are now connected through
a targeted augmentation artifact, but they remain sparse candidate targets
until they pass the same held-out factual gates as the core physiology.

## Dataset Inventory

| Dataset | Local location | Size | What it can support |
|---|---|---:|---|
| MIMIC-IV v3.1 | `/Users/chunyouchang/Downloads/mimic-iv-3.1` | 9.9G | ICU/hospital physiology, labs, medications, administrations, inputs/outputs, procedures |
| eICU 2.0 | `/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0` | 5.1G | Multi-hospital ICU physiology, nursing, respiratory, treatment and medication context |
| MIMIC-IV-ED 2.2 | `/Users/chunyouchang/Downloads/mimic-iv-ed-2.2` | 116M | ED triage/vitals/medications/diagnoses before ICU |
| MIMIC-IV notes | `/Users/chunyouchang/Downloads/mimic-iv-note-deidentified-free-text-clinical-notes-2.2 (2)` | 1.8G | Timestamped radiology and discharge text; structured observation extraction |
| MIMIC-IV Waveform | `/Users/chunyouchang/Desktop/llm_project/medical_jepa/physionet.org/files/mimic4wdb/0.1.0` | 13G | ECG, arterial pressure, pleth, respiration and related high-frequency signals |
| NHANES | Raw files not found locally | - | Only the existing cross-sectional audit report is available; raw reruns are currently unavailable |

## Directly Usable Now

### eICU transition library

The repository contains transition cohorts for 12 body-system modules. Most
have 1h, 3h, 6h and 12h views; AKI, sepsis, hepatic and heme modules also have
24h/48h views. They already follow the `*_t`, `*_tpN`, mask/age and action
context contract.

The largest useful cohorts are:

- AKI: 667,799 rows at 6h, 530,265 at 24h, 397,897 at 48h;
- sepsis: 422,244 rows at 6h, 328,759 at 24h, 238,649 at 48h;
- respiratory: 57,673 rows at 6h and more than 287,000 rows at 12h;
- endocrine stress: 17,009 rows at 6h;
- electrolyte/acid-base: 17,387 rows at 6h;
- heme/coagulation: 16,870 rows at 6h;
- neuro, cardiac, hepatic, GI/nutrition, skin/wound, rhabdo and toxic-metabolic
  transition sets.

These are the best current sources for factual routers, belief-state audits,
long-horizon observation and cross-system coupling.

### Full MIMIC-IV observation cohort

`/private/tmp/mimiciv_obs_6h.parquet` contains:

- 640,164 transition rows;
- 63,307 subjects;
- 90,190 ICU stays;
- 17 care units;
- 55 eligible numeric target variables.

The strict full-cohort audit validated:

- 19 same-time nowcast targets;
- 18 factual 6h forecast targets;
- 14 calibrated 90% forecast-interval targets.

This is currently the strongest single observation/prediction artifact.

### MIMIC-IV DKA cohort

`dka_transitions_6h_mimiciv_full_v31_icd.parquet` contains 10,785 DKA
transitions with glucose, pH, bicarbonate, anion gap, potassium, sodium,
osmolality, BHB, renal variables, vitals, urine output and treatment context.
It is the best source for the DKA JEPA / grey-box / factual-router branch.

## High-Value Data Not Fully Connected

### MIMIC-IV hormone labs

The MIMIC-IV lab dictionary contains the following hormone-related itemids,
and the full `labevents` table contains real observations for these items:

| Target | Itemid | Hospital rows | Subjects | ICU rows | ICU stays with data | ICU stays with repeated measurements |
|---|---:|---:|---:|---:|---:|---:|
| Cortisol | 50909 | 33,006 | 15,607 | 7,192 | 4,547 | 1,354 |
| PTH | 50965 | 63,748 | 20,401 | 980 | 851 | 83 |
| TPO antibody | 50992 | 5,893 | 5,147 | 143 | 139 | 4 |
| TSH | 50993 | 409,257 | 122,606 | 12,574 | 11,607 | 764 |
| Total T4 | 50994 | 25,802 | 13,247 | 1,868 | 1,530 | 176 |
| Free T4 | 50995 | 134,212 | 46,378 | 2,781 | 2,375 | 228 |

Interpretation:

- TSH and free T4 are the best candidates for hormone-axis nowcasting and
  longer-horizon factual analysis;
- cortisol is usable for stress-axis validation, but is not a dense vital sign;
- total T4 is usable but smaller;
- PTH is a sparse candidate, mainly for same-time or 24h research;
- TPO antibody is too sparse for a reliable dynamic forecast;
- growth hormone, insulin, renin and aldosterone itemids exist in the
  dictionary, but the current MIMIC `labevents` scan found no actual rows for
  them.

The transition extractor now maps cortisol, TSH, PTH, total T4, free T4 and
TPO antibody. The targeted augmentation artifact is ready, but the strict
audit did not promote any hormone target to factual nowcast or 6h forecast:
the limiting factor is paired transition support and measurement density, not
the absence of raw hormone rows.

### Administration-grade treatment context

MIMIC-IV has the strongest treatment data in the local machine:

- `icu/inputevents.csv.gz`: IV fluids, infusions, rates and amounts;
- `hosp/emar.csv.gz` plus `hosp/emar_detail.csv.gz`: medication administration;
- `icu/ingredientevents.csv.gz`: ingredients and rates;
- `icu/outputevents.csv.gz`: urine and other outputs;
- `icu/procedureevents.csv.gz`: procedures and device-related events;
- `hosp/pharmacy.csv.gz` and `hosp/prescriptions.csv.gz`: orders and medication
  context.

These can improve factual prediction under observed treatment. They do not
identify treatment effects and must not be used to make causal claims.

### eICU clinical context

eICU contains data that MIMIC-IV tables do not provide as directly:

- `nurseCharting.csv.gz`, `nurseAssessment.csv.gz`, `physicalExam.csv.gz`;
- `respiratoryCharting.csv.gz` and `respiratoryCare.csv.gz`;
- `note.csv.gz`, `treatment.csv.gz`, `infusionDrug.csv.gz`, `medication.csv.gz`;
- `intakeOutput.csv.gz`, `vitalPeriodic.csv.gz`, `vitalAperiodic.csv.gz`.

This is useful for timestamp-valid observation enrichment, respiratory state
and nursing-state extraction. The current eICU medication-derived transition
sets do not provide administration-grade dose resolution consistently, so
MIMIC inputevents/eMAR should be preferred for dose-sensitive modeling.

### ED and notes

MIMIC-IV-ED provides triage, repeated ED vitals, medication reconciliation,
Pyxis and ED diagnoses. It is directly usable for a pre-ICU observation layer
and ICU-entry baseline. ED-to-ICU incremental prediction remains candidate-only.

MIMIC notes currently include radiology and discharge notes, not nursing notes.
Radiology extraction is timestamp-safe and useful as observed evidence, but the
current physiology-to-radiology forecast audit passed 0/6 targets. It should
not be treated as a new numeric forecast source.

## Research-Only Assets

### Waveform data

The local waveform mirror contains about 15,100 files and 13G of data. The
aligned local cohort has 2,686 waveform segments across 171 stays / 170
subjects. All-target 1h/3h waveform sweeps evaluated 57 targets and promoted
zero waveform increments. Waveform remains candidate-only.

### Synthetic RCT data

`/private/tmp/osler_synthetic_rct_dka_100k.parquet` contains 6.6 million
synthetic rows and explicitly marks itself as fictional data. It is usable for
training-loop smoke tests, representation experiments and known-ground-truth
debugging. It is not clinical evidence and must not validate a real treatment
effect or open the causal gate.

### Checkpoints and caches

The `.pt`, `.npz` and belief-cache files are learned artifacts, not new patient
data. They can be evaluated or used as initialization under their own gates,
but they should not be counted as independent evidence or as additional
training subjects.

## Recommended Use Order

1. Keep the hormone augmentation and audit as candidate-only observation
   infrastructure; do not promote a hormone target without paired support and
   held-out improvement.
2. Add MIMIC `inputevents` + eMAR + ingredient/output context to factual
   forecasts, especially glucose, potassium, acid-base and fluid balance.
3. Use eICU nursing/respiratory tables for timestamped observation enrichment,
   with missingness and leakage guards.
4. Keep waveform and synthetic RCT branches candidate/research-only until they
   independently pass the existing heldout gates.

## Final Classification

| Data class | Decision |
|---|---|
| Full eICU transition cohorts | Use now for observation, factual prediction and belief audits |
| Full MIMIC-IV transition cohort | Use now; strongest cross-database observation source |
| MIMIC DKA transitions | Use now for DKA JEPA and grey-box work |
| MIMIC hormone labs | Use after target mapping; highest next-value expansion |
| MIMIC inputevents/eMAR | Use for observed-treatment factual context |
| ED tables | Use for pre-ICU observation and baseline |
| eICU notes/nursing/respiratory | Use for timestamped candidate observations |
| MIMIC notes | Use for timestamped evidence, not assumed forecast targets |
| Waveform | Candidate-only until a target passes the gate |
| Synthetic RCT | Training/debug only; never clinical evidence |
| Checkpoints/caches | Model artifacts, not data evidence |
| NHANES raw files | Not currently available locally; only prior audit artifact exists |
