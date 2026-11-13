# NWICU External Shared-State Validation

Date: 2026-09-01

## Purpose

Use the complete Northwestern ICU (NWICU) database as a new, independent
external lockbox for the frozen eICU shared-body-state methods. NWICU was not
used to choose the neural architecture, masking policy, PCA rank, sibling
guards, thresholds, or source routing.

This audit evaluates same-anchor reconstruction. The target, its complete
history, and near-deterministic sibling variables are hidden. The task asks
whether the rest of the measured body state can recover a fresh current
measurement. It is not a future forecast and is not causal.

## Native NWICU adapter

The new adapter uses NWICU's own `d_labitems` and `d_items` dictionaries. It
does not reuse MIMIC item IDs.

- Complete cohort: 27,956 ICU stays and 22,830 patients.
- Canonical ledger: 10,618,429 measurement events.
- Actual observed variables: 46.
- Derived events: 1,613,996 same-time MAP values, with arterial pressure
  preferred over non-invasive pressure.
- Whole-body examples: 428,689 anchors from 22,030 patients and 26,823 stays.
- Strict target labels: 2,034,064.
- The model-facing contract remains the frozen 58-variable eICU contract.
  Variables absent from NWICU remain missing rather than being fabricated.
- `Total CO2` is mapped to serum bicarbonate. Fahrenheit temperature and
  ounce body weight are converted to canonical units. CRP in mg/L is converted
  to mg/dL before plausibility filtering.
- NWICU has no `outputevents` or `inputevents` table. Urine and treatment
  channels are therefore absent in this audit. Treatment extraction is
  fail-closed rather than approximated from uncalibrated labels.

Aggregate extraction reports:

- `artifacts/nwicu_event_ledger_full_report_20260831.json`
- `artifacts/nwicu_event_examples_3h_full_report_20260831.json`

## Frozen nonlinear shared state

Training used the existing full eICU development examples, target-aware block
masking, fresh truth no older than one hour, and four reconstruction epochs.
Every target had to beat both the eICU population center and an equal-capacity
permuted-physiology placebo in all seven patient splits and the NWICU
care-unit gate.

| Target | NWICU rows | Patients | Shared MAE | Population MAE | Placebo MAE | Gain vs population |
|---|---:|---:|---:|---:|---:|---:|
| ALT | 97,059 | 6,620 | 5.628 | 5.694 | 5.690 | 1.15% |
| Bicarbonate | 278,605 | 20,962 | 0.814 | 0.875 | 0.883 | 7.04% |
| Total protein | 163,618 | 11,164 | 0.742 | 0.912 | 0.921 | 18.63% |

Artifact:
`artifacts/shared_state_reconstruction_strict_eicu_to_nwicu_20260901.json`.

ALT and bicarbonate reproduce the same nonlinear source-target relation that
previously passed the MIMIC-IV external audit. Total protein is a new NWICU
external pass.

## Frozen rank-32 physiology modes

The PCA rank remained fixed at 32, which had already been selected before
opening the NWICU results. Sixteen targets passed the same strict gate.

| Target | NWICU rows | Patients | Mode MAE | Population MAE | Placebo MAE |
|---|---:|---:|---:|---:|---:|
| Alkaline phosphatase | 196,920 | 12,706 | 1.471 | 1.481 | 1.481 |
| Anion gap | 280,306 | 21,020 | 1.219 | 1.301 | 1.264 |
| BUN | 379,652 | 20,961 | 1.045 | 1.088 | 1.096 |
| Chloride | 279,034 | 20,969 | 1.080 | 1.108 | 1.106 |
| Cortisol | 7,080 | 964 | 1.000 | 1.014 | 1.015 |
| Creatinine | 381,700 | 21,008 | 1.684 | 1.700 | 1.699 |
| Free T4 | 6,288 | 876 | 1.012 | 1.055 | 1.058 |
| Hematocrit | 358,241 | 20,295 | 0.799 | 0.837 | 0.884 |
| Lactate | 65,204 | 6,039 | 0.977 | 1.031 | 1.049 |
| MAP | 272,743 | 16,844 | 0.727 | 0.754 | 0.779 |
| Phosphate | 206,014 | 12,301 | 0.739 | 0.832 | 0.830 |
| Platelets | 366,414 | 20,670 | 0.976 | 1.000 | 0.997 |
| Potassium | 285,195 | 20,991 | 0.923 | 0.947 | 0.958 |
| Serum osmolality | 14,435 | 1,418 | 0.604 | 0.644 | 0.643 |
| Total protein | 163,618 | 11,164 | 0.636 | 0.912 | 0.908 |
| WBC | 365,803 | 20,652 | 1.100 | 1.117 | 1.117 |

Artifact:
`artifacts/low_rank_shared_state_rank32_strict_eicu_to_nwicu_20260901.json`.

Anion gap and phosphate reproduce the same rank-32 source-target relation that
previously passed the MIMIC-IV external audit. Cortisol and free T4 show that
the current shared state contains a small amount of recoverable endocrine
information; this does not establish irregular-event or future hormone
forecasting.

## Interpretation

The frozen eICU representation transferred to a second independent US ICU
database. Across MIMIC-IV and NWICU, four source-target relations now reproduce
without changing the method:

- nonlinear shared state: ALT and bicarbonate;
- rank-32 modes: anion gap and phosphate.

NWICU also produced additional source-specific external passes. They are valid
NWICU external results, but source rankings differ between databases. They
must not be combined into a target router selected after viewing NWICU and
then described as untouched validation. A combined runtime policy requires a
future independent lockbox such as HiRID or AmsterdamUMCdb.

The result supports a partial shared current body state, especially for
acid-base, mineral, renal-metabolic, perfusion, and blood-count structure. It
does not show that every target is reconstructable, does not validate shared
future dynamics, and does not allow treatment-effect claims.

## Reproduction

```bash
python -m extractors.canonical_event_ledger_extract \
  --database nwicu \
  --data-root /path/to/northwestern-icu-nwicu-database-0.1.0 \
  --output-dir /private/tmp/nwicu_event_ledger_full \
  --report artifacts/nwicu_event_ledger_full_report.json \
  --max-stays 100000 \
  --exclude-treatments
```

Build 3-hour examples with the exact development variable list, then run:

```bash
python -m scripts.validate_shared_state_reconstruction \
  --development-examples /path/to/eicu_3h_examples.parquet \
  --lockbox-examples /path/to/nwicu_3h_examples.parquet \
  --development-max-stays 100000 \
  --lockbox-max-stays 100000 \
  --epochs 4 \
  --max-truth-age-hours 1 \
  --bootstrap-samples 500 \
  --output artifacts/shared_state_eicu_to_nwicu.json

python -m scripts.validate_low_rank_shared_state \
  --development-examples /path/to/eicu_3h_examples.parquet \
  --lockbox-examples /path/to/nwicu_3h_examples.parquet \
  --development-max-stays 100000 \
  --lockbox-max-stays 100000 \
  --rank 32 \
  --max-truth-age-hours 1 \
  --bootstrap-samples 500 \
  --output artifacts/low_rank_state_eicu_to_nwicu.json
```

All row-level data and examples remain local. Only aggregate reports and
findings are suitable for version control.

## Multi-database state expansion

After opening the frozen single-source NWICU results, one broader candidate
was trained with eICU and MIMIC-IV as equal-weight development sources. Source
identity remained audit-only and was not provided to the physiology encoder.
The architecture, strict target-and-sibling mask, placebo, seven patient
splits and NWICU care-unit gate were unchanged.

The nonlinear shared state expanded from three to eight passing NWICU targets:

- liver/synthesis: ALT, AST, bilirubin and total protein;
- renal: BUN;
- inflammatory: CRP;
- acid-base/mineral: anion gap and phosphate.

Six targets were added, ALT and total protein remained, while bicarbonate was
lost. Therefore multi-source training broadens the represented physiology but
does not monotonically improve every target. The artifact is
`artifacts/shared_state_reconstruction_multisource_eicu_mimic_to_nwicu_20260901.json`.

A capacity hypothesis was also falsified. Increasing the same body latent from
64 to 128 dimensions produced seven passing targets, not eight, and still did
not recover bicarbonate. The wider branch was rejected and removed; its
aggregate negative-result artifact is
`artifacts/shared_state_reconstruction_multisource_latent128_to_nwicu_20260901.json`.

Because NWICU outcomes were already visible before this multi-source method
was evaluated, the eight-target result is an expanded candidate, not a new
untouched-lockbox promotion. Its next legitimate promotion test requires a
new database such as HiRID or AmsterdamUMCdb with all settings frozen.
