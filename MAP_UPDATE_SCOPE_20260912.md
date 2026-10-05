# MAP Update-Scope Diagnostic

Seven expanded full checkpoints. Original predictions reproduce bit-exactly.
Keep weights, query context and event tokens fixed; change active history
updates to observed MAP events only. Insulin-only and other-variable events
no longer update the filter. Synthetic test confirms inserted non-MAP/rate
events leave output unchanged under this diagnostic filter.

| MAP-only updates vs reference | Tune gain/harm | New source gain/harm | Hospital gain/harm |
|---|---:|---:|---:|
| Original full | 0/7,0/7 | 1/7,0/7 | 0/7,0/7 |
| Trained own | 0/7,2/7 | 0/7,4/7 | 0/7,0/7 |

No stable repair. The intervention removes event content as well as update
cadence, and weights were trained with full updates. Thus this is not proof
cadence is irrelevant, nor a trained MAP-update-only architecture evaluation.
It does not justify deploying skipped events. Same general intervention was
previously tested on glucose; this applies it to the expanded MAP trial.

All controls verified; no fitting, runtime, calibration or promotion changes.
Evidence: `artifacts/map_update_scope_20260912.json`.
