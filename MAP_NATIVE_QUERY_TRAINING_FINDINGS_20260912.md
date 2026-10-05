# MAP Native-Query Neural Training

## Seven-Split Completion

All seven own/full comparisons (14 fits) are now complete. Seven-split median
patient-equal MAE:

| Model | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.44082 | 11.73923 |
| Own MAP history | 8.84140 | 10.21067 |
| Full supported history | 9.22298 | 10.57510 |

Full-own has zero significant gains and seven significant harms in both
roles. Both arms beat persistence in all seven splits. Original seed7 is
unchanged. Summary verifies seed reports/checkpoint bindings; repeated hospital
tests are not independent replications. No clinical or runtime promotion.

Selected-checkpoint fit-gap replay reproduces all original evaluation/hospital
MAEs and intervals, plus selected tune minima. Median paired full-own MAE
differences: fit +0.23255, tune +0.25346, evaluation +0.38569, hospital +0.35316.
All four roles have 0/7 gains and 7/7 harms. Thus the selected full model is
already worse on the fit population; external shift is not the sole observed
stage of failure. Fit includes queries not sampled during optimization and
checkpoints are tune-selected. This is not proof that actual sampled examples
cannot be fitted, nor a unique optimization/architecture root-cause diagnosis.

New MAP timing alone is not a sufficient repair under this fixed training
policy. Do not repeat the same timing experiment as unfinished neural work.
Distinguishing sampled-fit performance from unvisited queries remains useful
before assigning a cause to representation or training dynamics.

Evidence: `artifacts/map_native_query_training_summary_20260912.json` and
`artifacts/map_native_query_fitgap_20260912.json`.

## Initial Seed Record

Seed 7 own/full fits completed under the frozen protocol in
`MAP_NATIVE_QUERY_TRAINING_PROTOCOL_20260912.md`. These use the new MAP
availability-native queries, not the old insulin-triggered query subset.
Both arms run 32 epochs; source-tune selects own epoch 25, full epoch 23.
Nine direct report bindings verify; seven batch/scheduler tests pass.

| Patient-equal MAP MAE | New source patients | Held-out hospitals |
|---|---:|---:|
| Persistence | 10.32894 | 11.73923 |
| Own MAP history | 8.80806 | 10.21067 |
| Full supported history | 9.16712 | 10.56384 |

Full minus own paired error: +0.35906 [0.21971, 0.49873] in 298 new source
patients; +0.35316 [0.04719, 0.66161] in 185 held-out hospital patients.
Both own and full significantly beat persistence. Thus this first split does
not support new MAP anchor timing alone as a repair of full-history benefit.

This is one split out of seven, not a stable failure verdict, complete gate or
proof that other-organ information is absent. Input content, update cadence,
representation and finite training budget remain possible contributors. The
current contrast alone does not distinguish them. Old/new query tasks differ,
so their unpaired MAEs cannot establish a timing-repair effect.

No runtime, interval or promotion change. Checkpoints and row predictions are
private. The initial six-splits-pending status is superseded above. Evidence:
`artifacts/map_native_query_training_7_20260912.json`.
