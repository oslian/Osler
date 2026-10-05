# Empty History Update Results

## Located Contract Issue

JEPA calls the existing NeuralBeliefFilter without its optional observation
mask. Entirely unobserved availability-history slots therefore change recurrent
state. A release-weight synthetic example with8 leading empty slots changes
state L2 by3.75717; the existing explicit mask gives exactly0 difference.
The new observed_steps candidate mode wires that existing mask through.
Physical zero remains observed; no event identity is inferred from values.

The three current query caches have zero duplicate(stay,time) anchors and
contiguous positional indexes. This does not constitute an exhaustive event
or historical-index audit.

Pre-source-support snapshot counts (all scored queries):

| Role/view | Empty steps / total | Queries with empty steps / total |
|---|---:|---:|
| development own |14440/71400|4867/7140|
| development whole |12827/71400|4642/7140|
| hospital own |14059/60770|4681/6077|
| hospital whole |12744/60770|4523/6077|
| MIMIC own |23706/66160|5847/6616|
| MIMIC whole |16025/66160|4635/6616|

## Matched Retraining

All28 prescribed fits completed, seven seeds,32epochs, same source data/splits,
sampler, objective and original numeric scale. Both own/whole arms change, so
unlike the dropout trial these are not expected to reproduce original logs.
Candidate metadata includes belief_input_mode; state_dict alone is insufficient.

Median patient-equal glucose@6h MAE, mg/dL:

| Arm | Hospital original | Masked | MIMIC original | Masked |
|---|---:|---:|---:|---:|
| Own bounded |41.322|41.475|34.415|34.905|
| Own identity |36.559|36.679|31.238|31.690|
| Whole bounded |42.388|42.481|35.572|35.913|
| Whole identity |38.202|38.070|33.368|33.497|

Paired significant improvements/harms vs same original arm:

- Own bounded: hospital0/0,MIMIC0/3.
- Own identity: hospital0/3,MIMIC0/4.
- Whole bounded: hospital0/3,MIMIC1/4.
- Whole identity: hospital0/0,MIMIC2/1.

Whole versus own improvement remains0/7 for both activations and domains.
Identity arms both beat persistence7/7 pooled, but TSICU remains0/7 for both.
The2014-2016 subgroup passes7/7 own and1/7 whole. These are descriptive
subgroups, not prospective time-heldout evaluations.

## Decision

The empty-update invariant is repaired in the candidate, but the matched trial
does not establish improvement or explain away the context transfer gap.
Do not turn a software-semantic correction into a prediction promotion.
Possible observation-process dependence is not proved to be the sole cause.
This does not test event-only recurrence or supervised future treatment changes.

300 tests pass; release validator remains41 catalog,27move,14fallback,
0missing,2derived on its bundled example. Default observed runtime behavior,
release weights and calibrators unchanged. No causal or untouched-lockbox claim.

Evidence: artifacts/encoder_history_mask_20260910.json,
artifacts/encoder_history_mask_comparison_20260910.json,
artifacts/encoder_history_mask_strata_20260910.json.
