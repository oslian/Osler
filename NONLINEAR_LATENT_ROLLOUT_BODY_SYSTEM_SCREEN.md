# Nonlinear Latent Rollout Body-System Screening

Audit date: 2026-07-12.

This is a **screening** run for the observation-only nonlinear latent rollout
candidate. It is not a promotion experiment: each module used 7 patient splits
and one hospital split, but only one training epoch and capped per-horizon
rows. The purpose is to find where the architecture is worth a full run and
where it is obviously unstable or uninformative.

All 15 local eICU body-system modules completed without a pipeline failure and
all future-column leakage checks passed.

| Module | 1h median delta | 3h median delta | 6h median delta | 12h median delta | Patient 168h stability |
|---|---:|---:|---:|---:|---:|
| acute_neuro | +0.058 | -0.001 | -0.003 | -0.010 | 4/7 |
| AKI | +0.165 | +0.010 | +0.006 | +0.003 | 2/7 |
| cardiac_injury | +0.135 | +0.002 | -0.000 | -0.002 | 3/7 |
| cardiovascular_instability | +0.104 | -0.000 | -0.011 | -0.028 | 5/7 |
| coagulopathy_heme | +0.128 | -0.001 | -0.003 | -0.016 | 6/7 |
| electrolyte_acid_base | +0.146 | +0.001 | -0.003 | +0.005 | 4/7 |
| endocrine_stress | +0.091 | -0.002 | -0.002 | -0.022 | 6/7 |
| GI/nutrition | +0.223 | +0.003 | +0.021 | +0.034 | 1/7 |
| hepatic_failure | +0.217 | +0.002 | -0.002 | -0.000 | 4/7 |
| immune_inflammatory | +0.070 | -0.030 | -0.034 | -0.009 | 4/7 |
| integumentary_skin_wound | +0.115 | +0.008 | +0.000 | -0.004 | 2/7 |
| musculoskeletal_rhabdo | +0.161 | +0.007 | -0.000 | +0.002 | 6/7 |
| respiratory | +0.114 | +0.026 | +0.004 | +0.019 | 4/7 |
| sepsis | +0.123 | +0.008 | +0.002 | -0.004 | 6/7 |
| toxic_metabolic | +0.070 | -0.004 | -0.010 | -0.008 | 7/7 |

Negative delta means lower MAE than persistence. Because this was a one-epoch
screening run, the small deltas around zero are not evidence of superiority.
The more reliable signal is architectural: 1h is usually persistence-dominated,
and several systems begin to show movement only after 3h or later.

## What this screen changes

1. The same model cannot be blindly promoted across all body systems.
2. Long-horizon stability is a first-class gate; GI/nutrition and AKI were
   particularly unstable in this low-budget screen.
3. The next full runs should prioritize modules with both a non-positive
   long-horizon delta and stable rollout: cardiovascular instability,
   immune/inflammatory, toxic-metabolic, AKI and sepsis.
4. Modules with sparse or proxy variables remain persistence/fallback until a
   full converged audit proves otherwise.

The complete per-module JSON reports are stored under
`/private/tmp/nonlinear_latent_rollout_*_screen.json`, with the aggregate at
`/private/tmp/nonlinear_latent_rollout_body_system_screen.json`.

No causal, counterfactual, clinical or runtime claim is allowed.
