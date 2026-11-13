# Projection optimization on real labels

All14 real-label fits completed.48bindings verified. Original native
evaluation/hospital scores reproduce within1e-7. Same labels/roles/events,
first observation Linear lr1e-3 instead of1e-4, other parameters unchanged.
Report: `artifacts/rate_anchor_jepa_real_fast_20260911.json`.

| Median patient MAE | Original native | Fast projection | Fast support |
|---|---:|---:|---:|
| Source evaluation | .387409 | .393918 | .393846 |
| Source bracket | .399871 | .401085 | .401055 |
| Hospital | .389022 | .394157 | .394236 |
| Hospital bracket | .426119 | .430363 | .430298 |

Fast real versus original native improves0/7allgroups; harms3/7source both,
0/7hospital all,1/7hospital bracket. Fast real versus fast support improves
0/7source both/hospital all,1/7hospital bracket; harm1/7source both/hospital all.
Fast versus own improves4/7source all,5/7bracket,0/7hospital both. Persistence
benefit7/7all remains. No stable repair or promotion.

## Diagnostic update

The same optimization change clearly recovers strong artificial event signal
but does not improve real-label forecasts. Thus successful synthetic pathway
learning does NOT establish this optimizer is the real bottleneck. Both real
and support worsen similarly, consistent with broader observation adaptation
rather than a specific new treatment-number benefit. Not a unique mechanism
proof, nor absence of weaker/conditional real information. Do not deploy this
learning-rate setting based on the artificial task.

This pair of experiments distinguishes a learnable architecture from a verified
clinical predictor: it can learn the constructed event relation; the captured
real event relation still lacks stable incremental value under the tested
contract. More blanket learning-rate/decoder changes are not justified by this
positive control alone. Treatment timing/semantics, conditional support and
physiological targets remain distinct uncertainties, not a proven data ceiling.

Seven splits overlap; no independent replication, new lockbox, calibration or
retention gate. Existing runtime weights, source registry and intervals unchanged.
