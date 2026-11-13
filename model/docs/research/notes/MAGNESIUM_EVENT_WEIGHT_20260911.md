# Future-event training weight diagnostic

Same old Mg queries, splits, fixed-lag tree features, source-tune selection,
and original patient-equal query evaluation. Changed only training weights:
each patient has equal total weight, each distinct stay/future-source event
within patient has equal weight, and its query weight is divided by multiplicity.
All three old control MAEs reproduce within1e-9 for every seed and role.
Reused future outcomes are legitimate different anchors, not asserted duplicates
or leakage; this changes the training emphasis rather than correcting a proven bug.

| Median MAE | Old own | Event-weight own | Old full | Event-weight full |
|---|---:|---:|---:|---:|
| Source evaluation | .219218 | .221100 | .224107 | .222177 |
| Hospital | .179137 | .179455 | .185421 | .182193 |

Event-weight full versus old full: significant gain/harm0/7 bothroles.
Event-weight full versus event-weight own: gain0/7, harm1/7 bothroles.
Event-weight own versus old own: source1gain1harm, hospital0gain1harm.
Event-weight full beats persistence7/7 both. Modest median shifts do not
establish repaired incremental whole-context generalization. This does not
test neural sampler changes or all weighting schemes. No runtime, calibration,
promotion or untouched external validation. Seven splits overlap.

Artifact: `artifacts/magnesium_event_weight_20260911.json`.
