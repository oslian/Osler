# MIMIC-IV Sparse Lab Irregular-Time Findings

Date: 2026-07-12

## What was fixed

The source parquet stored timestamps as `datetime64[us]`. The sparse audit was
using raw integer timestamps as if they were nanoseconds. That made time-window
selection wrong by a factor of 1,000. All timestamp arithmetic now passes
through `_time_ns`, and tests cover microsecond inputs.

The pair builder also previously kept only the first future lab. It now keeps
all future observations from 0.5 to 72 hours, then stratifies them by actual
elapsed time:

- 0--6 hours
- 6--24 hours
- 24--72 hours

The model receives elapsed time as an explicit forecast-query variable. It does
not receive the future lab value or post-anchor measurements.

## Results

| Target | Bin | Model | Pairs | Patient seeds | Care-unit result |
|---|---|---|---:|---:|---|
| INR | 6--24h | log event-history | 297,278 | 7/7 | significant |
| INR | 24--72h | log event-history | 546,472 | 7/7 | significant |
| PTT | 0--6h | log event-history | 81,993 | 7/7 | significant |
| Fibrinogen | 6--24h | raw event-history | 65,201 | 7/7 | significant |
| Cortisol | 0--6h | raw event-history | 1,778 | 7/7 | significant |

All other tested sparse targets were candidate/fallback. In particular,
troponin-T and bilirubin did not pass the full gate, and no direct insulin
sensitivity or hormone-axis hidden-state claim is made.

The follow-up conformal audit calibrated 90% intervals only for INR 6--24h
and INR 24--72h. PTT, fibrinogen, and cortisol remain point-only because their
interval coverage did not satisfy the full gate.

## Boundary

This is factual, observational prediction inside a single MIMIC center with
care-unit heldout validation. It is not causal, counterfactual, clinical, or
external-hospital evidence. The validated cells can be used as research/shadow
forecast sources; unsupported cells remain fallback.
