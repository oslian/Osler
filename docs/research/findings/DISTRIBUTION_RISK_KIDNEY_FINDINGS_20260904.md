# Distribution-Risk Kidney Router Audit

## Result

This audit does not learn a noisy realized-benefit label. The all-outcome target
expert predicts q05/q50/q95 of future delta. Persistence and the expert are then
compared by expected absolute loss under that conditional distribution.
All margins and interval scales are selected on eICU only.

| Target | Risk margin | eICU patient | eICU hospital | MIMIC-IV | Promotion |
|---|---:|---:|---:|---:|---:|
| creatinine@6h | 0.277349 | False | False | False | False |
| bun@6h | 2.08149 | False | False | False | False |
| urine_output@6h | -0.0771594 | False | False | False | False |
| potassium@6h | 0.316987 | False | False | False | False |
| bicarbonate@6h | 1.2077 | False | False | False | False |

## Boundary

This is a factual candidate audit. MIMIC-IV has been inspected and is not an
untouched lockbox. Failure leaves the release and persistence fallback unchanged.
No causal or clinical-decision claim is allowed.
