# Magnesium event-summary diagnostic

Same 2,366 eligible Mg queries and seven patient partitions as the matched
native experiment, in a potassium-selected cohort. Added existing 6/24-hour
event summaries to fixed-lag current/mask/age features. Available-time filtering
and source-version deduplication precede summaries; no future labels enter inputs.
Same source-tune tree grid, patient weights, and labels in five arms. All
recomputed lag-only source/hospital MAEs reproduce prior scores within 1e-9.

| Median MAE | Own event | Full event | Support event | Own lag | Full lag | Persistence |
|---|---:|---:|---:|---:|---:|---:|
| Source evaluation | .216566 | .221079 | .223434 | .219218 | .224107 | .277679 |
| Hospital | .179524 | .185237 | .180562 | .179137 | .185421 | .238434 |

Full versus own: significant gain 0/7 both roles, harm 1/7 source and 0/7
hospital. Full versus lag-full: gain/harm 0/7 both. Own versus lag-own:
gain 1/7 source, 0/7 hospital, no significant harm. Full versus support:
gain 0/7 both, harm 1/7 hospital. Own and full beat persistence 7/7 both.

Thus restoring these within-window summaries does not repair the incremental
whole-context gap. This is not a test of every possible event representation,
nor proof that raw events contain no useful cross-organ information. It is a
non-neural diagnostic, not retraining or replacement of shared JEPA. Seven
overlapping splits are not independent replications. No untouched external
database, interval calibration, runtime change, or promotion.

Two event-window unit tests pass. Private predictions remain outside Git.
Aggregate: `artifacts/magnesium_event_history_20260911.json`.
