# Source-Only Auxiliary Supervision Support

Question: can unchanged availability-safe glucose queries carry measured
future supervision for other variables? No new model or gate is evaluated.

Use exactly the original development patients plus the deterministic first2000
eligible extra source patients from the completed2250-fit experiment. Preserve
all original query anchors; analyze the original scored glucose queries only.
Do not load hospital or MIMIC labels for auxiliary selection.

For each canonical variable, current comes from the most recent measured result
available by anchor and age<=6h. Truth is the nearest draw to anchor+6h within
5-7h; equal distances choose earlier measurement then source ID. Final recorded
versions define retrospective labels only. Actual delta_t is retained. Values
are not interpolated. These are matched6h research labels, not a claim that
6h is the optimal biological horizon for every variable.

Current/future glucose must reproduce the established query dataset. Variable
support is counted on fitting patients separately for all seven original seeds,
with additional source patients always fitting and original tune unchanged.
An auxiliary is eligible for a later trial only if every fit split has at least
250 supported patients and1000 paired queries. This policy is frozen before
reading support results and does not establish forecastability. Report distinct
future draws separately from query pairs to expose label reuse.

Persist row-level arrays only in /private/tmp; Git receives aggregate support.
No clinical/causal claim, next-measurement prediction, runtime or weight change.
