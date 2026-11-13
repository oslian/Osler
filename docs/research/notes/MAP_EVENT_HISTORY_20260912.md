# MAP Event-History Repair Trial

## Question

Can quality-qualified 6/24h event summaries recover information omitted by
fixed history lags? Same MAP queries, measured labels, source fit/tune splits
and hospital roles as the preceding native and raw-history trials.

Use the existing event_windows helper with corrected quality handling. Mean,
standard deviation, min/max, last delta, slope and support summaries use only
as-of available source versions. Source-only channel support, patient-equal
weights and source-tune leaf selection remain fixed. Same tree hyperparameters
as the prior raw-history experiment. Five arms x seven splits x three leaf
settings; controls are retrained and predictions reproduce bit-exactly7/7.

## Results

Patient-equal MAE, median over seven seeds, mmHg:

| Model | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.71335 | 11.58096 |
| Fixed-lag own | 8.99978 | 10.39641 |
| Fixed-lag full | 9.42906 | 10.59785 |
| Events plus lag own | 9.13173 | 10.12921 |
| Events plus lag full | 9.16463 | 10.13827 |
| Events plus lag numeric-neutral support | 9.14217 | 10.22686 |

Common patient paired bootstrap,20,000draws, seed20260912:
full versus lag-full: gain2/7source4/7hospital, harm0/7both.
Own versus lag-own: gain0/7harm0/7both.
Full versus own and full versus support: gain0/7harm0/7both.
Full versus persistence: gain7/7harm0/7both.

## Interpretation and Limits

There is partial evidence that the fixed-lag representation was limiting the
full tree predictor. The hospital median reduction is about0.46mmHg. It is not
a universal repair and does not establish that non-target numeric physiology
adds value beyond own history or observation support. Summaries also change
feature count and support information; improvement cannot be assigned solely
to numerical excursions. No causal-organ claim follows.

This is not end-to-end JEPA training, conformal calibration, time-heldout
validation or an untouched external database test. No runtime promotion.
Raw patient data and saved per-query predictions remain private. Existing
release weights, registry, intervals and serving behavior are unchanged.

Evidence: `artifacts/map_event_history_20260912.json` and
`scripts/audit_map_event_history.py`. Parent dependencies verified before
training, private feature/prediction hashes bound in the report. Historical
event-summary reports are not retroactively declared quality-corrected.
