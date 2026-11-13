# Source Expansion Error Decomposition

Replayed every frozen small/expanded prediction on both cohorts, reproducing all
66 target/cohort MAE differences to1e-10. No weights, calibration, thresholds or
patient routes changed. Reused the tested exact error identity:

total MAE change = population-prior MAE change + change in model-minus-prior MAE.

| Cohort | Total change | Prior change | Conditional increment change |
|---|---:|---:|---:|
| Broad | -0.048489 | -0.004524 | -0.043965 |
| Selected | +0.012255 | +0.004615 | +0.007640 |

These are target-equal normalized errors, not clinical units. The decomposition
is algebraic, NOT a causal estimate of which training component caused gains.
Broad27/33 targets improve relative-to-prior error;30/33 beat their own prior in
pooled MAE. Selected15/33 improve relative-to-prior error;29/33 still beat their
own prior. These numeric counts do not establish corrected statistical passes.

Key distinctions:
- Broad AST gain-.137224 is mostly improved model-minus-prior error(-.132918),
  unlike ALT where most gain-.013919 comes from prior change-.011540.
- Broad troponin improves-.535434 but still loses its own prior by+.040043;
  lipase similarly improves-.187953 but remains+.007396 above prior. A smaller
  failure is not validated reconstruction.
- Broad BUN total+.000501 is slight harm despite conditional improvement-.004288,
  because its prior worsens+.004788. Do not label every total harm as lost state
  information.
- Selected glucose total+.095560 occurs despite a better prior(-.010220);
  conditional error worsens+.105779 and the model now loses its own prior.
- Selected AST conditional error worsens+.372038; troponin+.212687. These are
  not explained by simply using a worse population center.
- Selected BUN conditional error improves-.036155 while creatinine's tiny total
  gain-.001046 is prior-driven; conditional error actually worsens+.003241.

The remaining failure is heterogeneous, not a uniform offset fix. Source
expansion increases broad conditional usefulness but does not retain every
selected-population relationship. This does not prove gradient conflict,
memorization, irreducible noise, or absence of physiology. Group/site case mix,
sampling, selected epochs and source priors still differ together.

Next discriminate using source-only per-target training/selection trajectories,
including whether expanded training still fits selected patients but loses their
held-out counterparts. Do not derive a cohort router from retrospective insulin
selection or tune per-target weights on these reused evaluations. All39statuses
remain33trained/6pending; no new promotion or future/interval claim.

Evidence: artifacts/native_reconstruction_source_exposure_increment_20260919.json.
