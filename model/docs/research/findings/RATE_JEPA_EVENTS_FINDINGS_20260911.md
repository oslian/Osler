# Recorded treatment in the same JEPA history filter

Fourteen32epoch fits completed (seven seeds, real/support-only event channels).
Same existing filter, zero-expanded observation projection, unchanged lab step
mask and clock. Initial predictions reproduce within1e-5 on checked tune rows.
Two component tests verify clock preservation, zero-initial equivalence,
nonzero event-weight gradient, source-only scaling and zero/missing separation.

| Median patient MAE | Original whole | Real event history | Event support |
|---|---:|---:|---:|
| Source all | .384183 | .384201 | .384197 |
| Source bracket-after | .393479 | .394020 | .393688 |
| Hospital all | .386973 | .387170 | .387290 |
| Hospital bracket-after | .413968 | .413986 | .413953 |

Real versus original: significant gain1/7source,0/7hospital, harm0/7, both
groups. Real versus event support: gain0/7every group; hospital harm1/7both.
Persistence beaten7/7every group. All selected epochs match the original whole
arm. Predictions change little; no reliable numerical treatment increment.

This does not establish that treatment is irrelevant. Only retrospective
recorded insulin history was added, at existing laboratory-supported update
steps. Treatment-only updates, verified availability, delivered dose, other
medications and a persistent organ-state model remain outside this contrast.
Further diagnosis must inspect effective use of the event path, not assume
nonzero component-test gradients imply useful whole-model sensitivity.

Aggregate `artifacts/rate_anchor_jepa_events_20260911.json`. No runtime,
interval or promotion changes. The original run binds the base training
protocol and runner; event protocol existed before execution, but was not
included in its binding map. Subsequent runner revision adds that metadata
binding without altering training mathematics. Original report preserved.
