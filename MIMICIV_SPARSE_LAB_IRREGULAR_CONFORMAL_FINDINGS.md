# Sparse Irregular-Time Conformal Findings

Date: 2026-07-12  
Interval level: 90% split-conformal residual interval

## Gate

Each validated irregular-time point forecast was split into fit,
calibration, and test groups. The calibration residual radius was learned
without using test values. The interval was accepted only when coverage was
0.87--0.93 on all seven patient-heldout seeds and on the internal care-unit
heldout split.

## Results

| Target/bin | Patient coverage passes | Care-unit coverage | Interval status |
|---|---:|---:|---|
| INR 6--24h | 7/7 | 0.8937 | calibrated |
| INR 24--72h | 7/7 | 0.9098 | calibrated |
| PTT 0--6h | 7/7 | 0.9594 | needs calibration audit |
| Fibrinogen 6--24h | 7/7 | 0.9555 | needs calibration audit |
| Cortisol 0--6h | 5/7 | 0.8968 | needs calibration audit |

PTT and fibrinogen intervals are too wide under the current residual model,
and cortisol is not stable across all patient splits. They remain point-only
research forecasts with no numeric interval in the unified contract.

## Boundary

INR intervals are calibrated for the specified irregular-time query bins in a
single MIMIC center with care-unit heldout validation. This is observational
factual uncertainty calibration, not causal, counterfactual, clinical, or
external-hospital evidence.

