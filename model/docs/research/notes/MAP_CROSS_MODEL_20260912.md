# MAP Cross-Model Matched Scores

Saved predictions only, no fitting/selection. Same eligible labels, current
values and patient partitions; all parent bindings verified. Common20k patient
bootstrap. The tree own-summary arm still includes full fixed-lag context;
its name means only the appended summaries are own-only.

| Contrast | Source gain/harm | Hospital gain/harm |
|---|---:|---:|
| Tree own-summary vs own JEPA | 0/1 | 0/0 |
| Tree full-summary vs own JEPA | 0/1 | 0/0 |
| Tree own-summary vs full JEPA | 1/0 | 0/0 |
| Tree full-summary vs full JEPA | 0/0 | 0/0 |

Each count is out of seven. Hospital median paired error difference for tree
full-summary minus own JEPA is -0.11373mmHg; no seed establishes significance.
Thus the documented improvement over the older lag-tree does not establish
superiority over existing own-history JEPA. It cannot justify replacing that
model or claim JEPA failed to retain all useful temporal information.

Inputs differ (native event stream plus insulin for full JEPA versus fixed
lags and summaries for tree), along with learning algorithm and capacity.
This is an outcome-level comparison, not an encoder-only intervention, and
nonsignificance does not prove equivalence. No intervals, runtime, promotion
or new external lockbox. Evidence: `artifacts/map_cross_model_20260912.json`.
