# Utility-Conditioned Kidney Router Audit

## Result

The router is supervised by out-of-sample realized benefit rather than future-change magnitude.
A target expert is fit on one patient partition; the router is fit on disjoint patients.
MIMIC-IV labels do not select the router, cutoff or interval scale.

| Target | Cutoff | eICU patient | eICU hospital | MIMIC-IV | Promotion |
|---|---:|---:|---:|---:|---:|
| creatinine@6h | 0.500 | False | False | False | False |
| bun@6h | 0.500 | False | False | False | False |
| urine_output@6h | 0.500 | False | False | False | False |
| potassium@6h | 0.500 | False | False | False | False |
| bicarbonate@6h | 0.500 | False | False | False | False |

## Boundary

This is a factual candidate audit. Failure leaves the validated release and persistence fallback unchanged.
MIMIC-IV has been inspected previously and is not described as an untouched lockbox.
No causal or clinical-decision claim is allowed.
