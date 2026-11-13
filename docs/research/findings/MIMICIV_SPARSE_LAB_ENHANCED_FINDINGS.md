# Sparse Laboratory Enhanced Model Findings

Date: 2026-07-12  
Scope: MIMIC-IV ICU factual observation only

## Experiment

Compared three sources:

1. persistence;
2. raw event-history ridge using current values, previous marker value,
   previous slope, observation interval, and observation count;
3. log-space event-history ridge for positive heavy-tailed markers.

All features were computed from observations at or before the anchor. The
candidate was required to beat persistence on seven patient-heldout seeds and
the internal care-unit heldout split.

## Important correction

An initial within-window implementation appeared to validate log-space INR and
PTT at 6/24/48 hours. The same next lab was reused across horizons, so those
were not genuine endpoint forecasts and were rejected.

The corrected endpoint audit requires the target measurement to occur in
0.75h--1.25h around the requested horizon, with a 0.5h minimum. Results:

- troponin-T: 0 endpoint pairs at 6/24/48h;
- INR: 25 endpoint pairs at 6h, 0 at 24/48h;
- PTT: 26 endpoint pairs at 6h, 0 at 24/48h;
- fibrinogen: 0 endpoint pairs;
- bilirubin: 35 endpoint pairs at 6h, 0 at 24/48h;
- cortisol and TSH: 0 endpoint pairs;
- validated cells: **0**.

## Extraction bug fixed and irregular-time result

The MIMIC parquet cache used `datetime64[us]` while the first implementation
treated integer timestamps as nanoseconds. This expanded search windows by
1,000x. All sparse audit time arithmetic now normalizes timestamps to int64
nanoseconds, with regression tests covering microsecond input.

The corrected irregular-time audit also keeps **all** future observations from
0.5 to 72 hours instead of only the first future lab. It treats elapsed time as
an explicit forecast query and reports three bins:

| Target | Time bin | Model | Pairs | Status |
|---|---|---|---:|---|
| INR | 6--24h | log event-history ridge | 297,278 | validated internally |
| INR | 24--72h | log event-history ridge | 546,472 | validated internally |
| PTT | 0--6h | log event-history ridge | 81,993 | validated internally |
| Fibrinogen | 6--24h | raw event-history ridge | 65,201 | validated internally |
| Cortisol | 0--6h | raw event-history ridge | 1,778 | validated internally |

The internal gate was seven patient-heldout seeds plus a care-unit heldout
split. These are not external hospital validations, causal effects, hormone
axis claims, or clinical predictions. Troponin-T and bilirubin had irregular
pairs but did not pass; TSH and the other sparse hormone targets remained
candidate/fallback.

## Conclusion

The log transform and patient-specific event history are useful for selected
irregular-time coagulation and cortisol targets after the timestamp bug was
fixed. They do not make every sparse target predictable. Unsupported targets
remain fallback/candidate-only. This is a measurement schedule limitation, not
evidence that the underlying biology is unmodelable.
