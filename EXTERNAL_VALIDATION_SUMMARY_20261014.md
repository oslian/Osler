# External Validation Summary: Value-Conditional Intervals on eICU and NWICU (2026-10-14)

Detail reports:
`EXTERNAL_eICU_VALUE_CONDITIONAL_INTERVAL_49CELL_FINDINGS_20261014.md`,
`EXTERNAL_NWICU_VALUE_CONDITIONAL_INTERVAL_49CELL_FINDINGS_20261014.md`,
protocol `NWICU_EXTERNAL_VALIDATION_PROTOCOL_20261014.md`.

## NWICU protocol outcome (primary endpoint)

- Evaluable: 40/49. Excluded by design: 3 urine-output cells and 6 treatment-context cells.
- Point gate (patient axis; single hospital): 40/40.
- **Full pass with the primary value-conditional interval: 26/40.**
- Secondary fixed-width interval: also 26/40.
- The value-conditional interval gained 7 cells (`respiratory_rate@6h`, `creatinine@12h`,
  `hemoglobin@12h`, `respiratory_rate@12h`, `bun@24h`, `glucose@24h`, `creatinine@48h`) and lost
  7, **all by over-coverage above 0.93** (`heart_rate@3h`, `hematocrit@3h`, `hemoglobin@6h`,
  `heart_rate@12h`, `hematocrit@12h`, `creatinine@24h`, `heart_rate@24h`).

## eICU (MIMIC v2 data, all 49 cells)

- Point gate (patient and hospital): 33/49.
- Full pass: fixed 12 → value-conditional **21**. The value-conditional interval gained 11 cells
  and lost 2 (`heart_rate@24h`, `heart_rate@48h`, both by over-coverage).

## What the two databases show together

1. **Case-mix shift is the main reason the fixed interval fails on eICU.** Across the same 36
   continuous cells, median fixed-width patient coverage is 0.861 on eICU, whose cohorts come
   from disease modules, and 0.897 on NWICU, a whole-ICU cohort like MIMIC. Where case mix
   matches, the fixed interval already transfers.
2. **The value-conditional interval fixes case-mix shift but is systematically conservative.**
   MIMIC patient-fold coverage is 0.902–0.944 (median 0.915) against a 0.90 target. That is a net
   gain on eICU and a wash on NWICU, where the extra width pushes 7 cells above 0.93.
3. **15 cells pass both external databases under the primary method:**
   `hypoxemia_event@1h`, `hypoxemia_event@3h`, `map@3h`, `respiratory_rate@3h`, `heart_rate@6h`,
   `respiratory_rate@6h`, `hemoglobin@12h`, `map@12h`, `potassium@12h`, `bicarbonate@24h`,
   `bun@24h`, `glucose@24h`, `potassium@24h`, `sodium@24h`, `creatinine@48h`.

## Data drift (frozen 2026-08-16 data vs v2 rebuild)

The current extractor reproduces every cell's MIMIC row count except urine output (−1.4% to
−2.6%, urine derivation changed). Treatment `hist_*` columns differ in part. On eICU the point
gate changed in 5 cells (`platelets@1h`, `hemoglobin@3h`, `urine_output@3h`, `urine_output@6h`
lost; `bicarbonate@12h` gained). Fixed-width full passes went from 11 (frozen) to 12 (v2).

## Execution notes

- Runs were split by horizon to fit in 16 GB of memory and merged with
  `merge_external_value_conditional_runs.py`. No method or gate changed.
- The first NWICU invocation failed in the shell driver: zsh did not split the `--protocol`
  argument, and argparse rejected it before the scoring script ran. No NWICU label was read and
  no output was written. The corrected invocation ran once with protocol hashes verified.

## Decision

- Under the frozen protocol, the 26 NWICU full-pass cells are **eligible for promotion review**.
  Nothing is promoted automatically, and the runtime registry is unchanged. The 15 cells that
  pass both eICU and NWICU are the strongest candidates.
- The over-coverage of the value-conditional interval is a method issue. Any correction is a new
  method version: it must be developed without NWICU labels and validated on another untouched
  database (for example HiRID or AmsterdamUMCdb). NWICU is now used.

`causal_claim_allowed: false`, `clinical_promotion_allowed: false`.
