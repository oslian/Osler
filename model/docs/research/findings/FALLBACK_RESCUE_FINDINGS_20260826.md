# Fallback Cell Rescue Findings (2026-08-26)

## Question

Can additional release cells move beyond persistence without weakening the
existing patient, care-unit, and forward-time gates?

## Method

Three unresolved 6h cells with locally available raw MIMIC-IV labels were
tested using a new eICU-only discovery method:

- hospital-balanced ridge residual fitting;
- discovery-only shrinkage and selective abstention;
- a capacity-matched shuffled-label placebo;
- for hematocrit and hemoglobin, a pre-anchor `Hct ~= 3 * Hgb` consistency
  gate selected only on eICU;
- MIMIC-IV used only for the final replication score.

The release registry and weights were not modified unless every patient,
care-unit, time, and placebo gate passed.

## Results

| Cell | MIMIC patient splits | Forward time | Care unit | Result |
|---|---:|---:|---:|---|
| creatinine@6h | 0/7 | fail | fail | rejected |
| hematocrit@6h | 7/7 | pass | fail | near-miss, not promoted |
| hemoglobin@6h | 0/7 | fail | fail | rejected |

For `hematocrit@6h`, the consistency-gated model reduced row-level MIMIC MAE
from `2.3667` to `2.2872`, and its forward-time patient-equalized CI remained
below zero. The care-unit CI was `[-0.1020, 0.3474]`, so the improvement was not
stable enough across care settings.

`creatinine@6h` and `hemoglobin@6h` reversed direction on MIMIC-IV. Their eICU
residual signal therefore does not transfer reliably under the current input
contract.

## Decision

No new continuous cell is promoted. The release remains at 27 movable cells
and 14 persistence cells. `hematocrit@6h` is retained only as a research
candidate; formal promotion requires a fresh untouched database such as HiRID
or AmsterdamUMCdb, not another MIMIC-tuned rescue.

This remains factual research only. `causal_claim_allowed` and
`clinical_promotion_allowed` are both false.
