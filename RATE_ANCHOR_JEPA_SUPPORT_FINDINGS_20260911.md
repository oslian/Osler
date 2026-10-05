# Train-matched support control result

Seven end-to-end support-control fits completed under the fixed protocol in
`RATE_ANCHOR_JEPA_SUPPORT_PROTOCOL_20260911.md`. Original own/whole predictions
were hash verified; no reference weights or runtime artifacts were changed.
Aggregate: `artifacts/rate_anchor_jepa_support_20260911.json`.

| Median patient-equal MAE | Own | Whole numeric | Support only | Persistence |
|---|---:|---:|---:|---:|
| Source evaluation, all | .417130 | .384183 | .394297 | .496675 |
| Source evaluation, bracket-after | .426164 | .393479 | .403997 | .508829 |
| Hospital, all | .402915 | .386973 | .400350 | .482238 |
| Hospital, bracket-after | .437972 | .413968 | .430569 | .523661 |

Support-only versus own significantly improves 5/7 source comparisons in both
groups, but 0/7 hospital comparisons. Whole versus support significantly
improves 1/7 source comparisons in both groups, 2/7 hospital all-window and
0/7 hospital bracket-after comparisons. No significant harm in these contrasts.
Support-only beats persistence7/7 in every group. Seeds overlap in patients.

## Diagnosis

The source own-to-whole improvement is not uniquely attributable to other lab
numbers: a trained model without those numbers also gains. The support/age
representation and associated context path can carry predictive associations.
This does not distinguish clinical severity from institutional measurement
practice, nor assign an additive fraction of the full gain to either.

Unlike frozen input neutralization, this control can adapt during training.
Its results qualify the previous numerical sensitivity evidence: beneficial
dependence on numbers in fixed weights is not the same as stable incremental
benefit over a model trained without those numbers. Whole remains numerically
better, with partial statistical evidence, not equivalence or absence of signal.

The first transport failure for the support-only increment is now isolated:
source-patient gain5/7 becomes hospital gain0/7. Numerical cross-context also
does not consistently clear the matched control. That is a narrower failure
than "JEPA cannot learn", but it does not yet establish a unique causal
explanation or prove that removing support improves prediction. Hospital
support remains limited and reused; no new independent database was tested.

Do not remove truthful masks, claim missingness is meaningless, or deploy the
support model as a repair. Keep the current runtime. A repair must improve the
same paired future predictions, not merely pass a structural invariant.
Treatment-conditioned state updates and persistent organ-state dynamics are
not tested by this lab-only base JEPA control. No intervals or promotion.
