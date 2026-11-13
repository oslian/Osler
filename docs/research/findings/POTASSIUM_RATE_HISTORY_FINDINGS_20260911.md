# Same-label rate history: partial own-history gain only

28 source-only fixed fits completed. Same6174pairs/1309patients and frozen
lab baseline reproduction; protocol POTASSIUM_RATE_HISTORY_PROTOCOL_20260911.md.
Compatible same-label rate sequences support341insulin and16diuretic windows;
other drug-compatible rate sequences havezero support. Solution flow is not
drug rate, and no delivered-dose integration is inferred.

Median patient-equal MAE:
own .373470 -> .369221; full .369971 -> .369971; persistence .437907.
Own real vs baseline significant gain1/7; vs shuffled1/7. Full vs baseline
or shuffled0/7. Both vs persistence7/7. Counts are unadjusted exploratory
comparisons on overlapping splits, not independent replication/promotion.

Thus recovering rate range/change/span does not establish stable whole-context
gain. It does not test a full event encoder, actual cumulative dose or all
therapies. Negative result for unsupported channels is not an efficacy test.
Future event exclusion, physicalzero and cross-label ambiguity have regression
coverage. Event-time remains retrospective; independent availability absent.

Artifact: `artifacts/potassium_rate_history_20260911.json`.
No external data, runtime weights, calibration or promotion change.
