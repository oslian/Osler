# Clean native change-size diagnosis

Source-fit patient-weighted75th percentile absolute glucose delta defines
descriptive larger/smaller groups (cutoffs65-71mg/dL). Group membership uses
future truth and is NEVER a prospective selector or clinical risk threshold.
Saved predictions only; no fitting or gate changes.

Clean-full versus clean-own significant gain/harm out of7:

| Group | Fit | New source patients | Hospital |
|---|---:|---:|---:|
| Larger actual change |7/0|1/2|0/1|
| Smaller actual change |4/1|0/3|1/1|

Hospital larger group90-94patients; patient-equal mean absolute true delta
median170.81 versus full predicted117.20. Smaller group152-155patients,
true28.89 versus predicted36.41. These absolute-magnitude summaries do not
measure conditional bias or prove that multiplying deltas fixes errors.
Future heterogeneity can make conservative forecasts optimal under MAE.

The fit-only context benefit also occurs for larger changes and fails to
transfer; this is not solely stable cases overwhelming the metric. Both
attenuation of large changes and movement on smaller changes remain. No
new interval, runtime or promotion, and no proof of irreducible uncertainty.
Artifact: `artifacts/clean_glucose_change_20260911.json`.
