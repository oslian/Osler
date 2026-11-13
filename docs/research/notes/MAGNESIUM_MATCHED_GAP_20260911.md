# Matched magnesium gap localization

Saved whole/own predictions evaluated without refitting or changing selection.
All parent hashes verified, roles patient-disjoint, heldout scores reproduced,
and saved whole source-tune MAE matched training selection exactly.

| Role | Own median MAE | Whole median MAE | Whole significant gain /7 |
|---|---:|---:|---:|
| Fit | .200352 | .179238 | 6 |
| Tune | .218106 | .222762 | 0 |
| Evaluation | .217013 | .218283 | 0 |
| Hospital | .182700 | .184960 | 0 |

No significant whole-own harm in these four roles. Fit confidence intervals
are descriptive in-sample comparisons, not evidence of generalization.
The extra-context advantage is already absent on source-tune patients, before
hospital transfer. This reproduces the location previously observed for glucose
under a different magnesium-primary loss, and argues against hospital shift or
glucose priority as sufficient explanations. It does not isolate spurious
correlations, optimization, representation or sampling as the unique cause.
No proof of absent cross-organ information or equivalence. No new calibration,
runtime change, promotion or fresh external validation.

Artifact: `artifacts/magnesium_matched_gap_20260911.json`.
