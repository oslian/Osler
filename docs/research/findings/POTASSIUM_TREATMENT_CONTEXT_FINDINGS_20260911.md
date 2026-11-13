# Treatment context: no stable incremental repair

Completed28 fixed-capacity source refits, seven original patient partitions,
6174 potassium pairs from1309patients. Frozen own/full event-window baselines
reproduce before comparisons. No external data or runtime changes.
Protocol: `POTASSIUM_TREATMENT_CONTEXT_PROTOCOL_20260911.md`.
Aggregate: `artifacts/potassium_treatment_context_20260911.json`.

Patient-equal MAE, median across overlapping source splits:

| Input | Baseline | Plus real treatment | Plus shuffled treatment |
|---|---:|---:|---:|
| Own lab history | .373470 | .372184 | .374051 |
| Full lab history | .369971 | .369971 | .371022 |

Persistence median .437907. Real treatment vs baseline: own1/7significant
improvement,1/7harm; full0/7improvement,0/7harm. Real vs shuffled0/7improvement
both modes. Both real arms beat persistence7/7; that is not incremental
treatment evidence. No stable diagnostic pass or promotion.

## Did the trees actually use the added input?

Read-only inspection of saved sklearn predictor nodes: real-treatment splits
for own by seeds7/19/31/43/59/71/83 are41/165/61/21/29/37/42. Corresponding
full counts0/0/2/0/3/3/7. Three full fits ignore every treatment feature and
reproduce baseline predictions; others use only a few insulin features.
Own models also predominantly use insulin, with a diuretic count in seed31.
No real arm uses potassium-repletion features in a split.

This localizes non-use under this fixed tree objective/capacity, NOT biological
irrelevance. Only20queries had potassium solution flow, without inferred drug
concentration. Coverage, min-leaf support and competition with existing lab
predictors constrain what this experiment can learn. Do not claim these
individual explanations have been experimentally separated.

## Boundaries

- This is retrospective infusion event-time context. Infusionoffset lacks
  independent available-time evidence; it cannot qualify as prospective use.
- No recorded event is not no treatment. Unknown rates remain missing, zero
  remains observed; solution flow is never relabeled as drug delivery.
- Latest6h snapshots are not full cumulative delivered dose or intent.
- Placebo donor histories stay within fit/tune/evaluation and before recipient
  relative time. Early unmatched donor queries remain missing. Thus donor
  coverage may differ; placebo is not exact missingness-matched evidence.
- Leaf count was inherited from the lab-only model, not tuned for treatments.
  Failure does not rule out all treatment-aware temporal architectures.
- No new pristine data, causal claim, calibrator or runtime source added.

The input omission is real, but simply appending these sparse snapshots does
not resolve incremental whole-context instability. A complete delivered-dose
or event-encoder claim remains untested; do not repeat this snapshot append
as a new solution. Exact availability and delivered-dose support need evidence,
not assumptions. This remains a diagnostic comparator, not a second deployed
body model.
