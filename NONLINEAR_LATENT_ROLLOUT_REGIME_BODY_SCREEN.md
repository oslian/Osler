# Regime/Uncertainty Whole-Body Screening

Audit date: 2026-07-12.

This is a low-budget observation-only screen after adding three-regime latent
dynamics, history length 9 and MC-dropout uncertainty gating. All 15 local
eICU body-system modules completed; all future-column leakage checks passed;
no module was promoted.

The screen used 7 patient splits, one hospital split, one training epoch and
capped rows. It is therefore a routing screen, not final accuracy evidence.

| Module | Patient 168h bounded | Hospital bounded | Median delta over horizons >=3h |
|---|---:|---:|---:|
| acute_neuro | 6/7 | no | -0.011 |
| AKI | 5/7 | yes | -0.046 |
| cardiac_injury | 4/7 | yes | -0.004 |
| cardiovascular_instability | 5/7 | yes | -0.014 |
| coagulopathy_heme | 4/7 | no | -0.015 |
| electrolyte_acid_base | 7/7 | yes | -0.017 |
| endocrine_stress | 6/7 | no | -0.029 |
| GI/nutrition | 1/7 | no | -0.002 |
| hepatic_failure | 6/7 | yes | -0.012 |
| immune_inflammatory | 4/7 | yes | -0.052 |
| integumentary_skin_wound | 5/7 | yes | -0.008 |
| musculoskeletal_rhabdo | 1/7 | no | -0.002 |
| respiratory | 6/7 | no | -0.013 |
| sepsis | 5/7 | yes | -0.041 |
| toxic_metabolic | 5/7 | no | -0.021 |

The main result is architectural rather than a universal score: the model
needs per-system stability gates. The same nonlinear transition cannot be
trusted for GI/nutrition or rhabdomyolysis just because it works for AKI or
sepsis. These modules remain persistence/fallback until a converged,
independent audit passes.

Aggregate report:
`/private/tmp/nonlinear_latent_rollout_body_system_regime_screen.json`.

No causal, counterfactual, clinical or runtime claim is allowed.
