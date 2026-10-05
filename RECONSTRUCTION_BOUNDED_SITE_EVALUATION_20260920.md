# Bounded Site Frozen Evaluation

Both source-selected epoch7 checkpoints frozen. Same83874 broad and4637
selected patient-target pairs as previous evaluations; exact truth identity,
complete unique pairs and shard hashes checked. All old reference MAEs reproduce
within1e-10 and11 output bindings verified. No evaluation-driven checkpoint
replacement, target routing, weighting or threshold adjustment.

| Cohort | Uniform replacement | Bounded site replacement | Relative change | Numerically better targets |
|---|---:|---:|---:|---:|
| Broad | 1.277458433 | 1.280860946 | +.266% | 17/33 |
| Selected | 2.918501095 | 2.909265049 | -.316% | 20/33 |

Target-equal normalized MAE, not clinical accuracy percentages. Both are worse
in macro than previous continued model1.265322/2.893127; that comparison changes
sampling and training exposure/budget and is not the isolated site intervention.

| Target | Broad uniform -> bounded | Selected uniform -> bounded |
|---|---:|---:|
| Creatinine | 1.474494 -> 1.444344 | 1.434976 -> 1.426672 |
| BUN | .898790 -> .958874 | .899127 -> .943422 |
| Glucose | 1.000172 -> .988005 | 1.755859 -> 1.796806 |
| Lipase | 1.679289 -> 1.775111 | 1.397890 -> 1.473515 |
| Troponin I | 5.823638 -> 5.862270 | 38.193719 -> 38.203378 |

Zero targets pass the existing source-support + corrected patient/site paired
gate against matched uniform sampling, in either cohort. Zero pass all reported
diagnostic reference comparisons. New matched process/placebo were not trained,
so these are not complete promotion evaluations. Six untrained heads remain
explicit and must not be counted as failed trained models.

## Decision

The fixed bounded-site sampling repair does not solve generalization. Its
source-selection macro improvement1.14% failed to carry over consistently.
Creatinine improves both but BUN/lipase worsen both, and glucose trades groups.
Site imbalance is real but not established as the principal remaining cause.
Do not retain only successful targets or tune weight caps using these outcomes.
Do not repeat this exact [.5,2] intervention as untested. Source-selection
uncertainty and target-specific response remain unresolved, not proof that
all site-aware training is impossible.

Evidence: artifacts/native_reconstruction_bounded_site_evaluation_20260920.json.
Current-value masked reconstruction only; reused cohorts, not an untouched
external lockbox, futureforecast, causal test or runtime promotion. No release
weights, registry, intervals or serving changes.
