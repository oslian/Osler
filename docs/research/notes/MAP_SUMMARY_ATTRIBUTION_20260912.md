# MAP Summary Attribution Follow-up

No fitting. Reuse the bound predictions from the MAP event-history trial;
same labels, patient roles and common20k patient bootstrap.

Numeric-neutral support model versus old lag-full:
- Source evaluation: gain2/7,harm0/7; median paired MAE difference -0.25940.
- Hospital: gain4/7,harm0/7; median paired difference -0.47568mmHg.

Support versus event-augmented own model: gain0/7,harm0/7both;
median paired difference +0.04226source,+0.06111hospital.

Therefore the prior full-versus-lag-full improvement does not uniquely support
non-MAP numerical excursions as its explanation. A model without those numbers
also improves that old baseline. The control still contains MAP numerical
summaries AND other-variable support summaries, so their contributions are not
separated by this comparison. Do not claim workflow-only learning, no useful
physiology, or equivalence based on a nonsignificant CI. Changing feature count
and removing old non-MAP numbers are also part of this contrast.

Evidence: `artifacts/map_summary_attribution_20260912.json`,
`scripts/audit_map_summary_attribution.py`. No runtime, weights, calibration or
promotion changed. This qualifies the partial-repair interpretation rather
than announcing a new candidate.
