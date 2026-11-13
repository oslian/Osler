# MAP Native Query Value-Objective Check

Recomputed the exact SmoothL1 formula (beta=1 in inherited normalized MAP
coordinates) on saved selected-checkpoint predictions. Patient-equal averaging
matches the patient-balanced population objective, not the finite sampled
training trajectory. No predictions, weights or selection changed.

| Group | Median full-own SmoothL1 difference | Full numerically worse |
|---|---:|---:|
| Fit population | +0.008385 | 7/7 |
| Tune | +0.011885 | 7/7 |
| New source patients | +0.016934 | 7/7 |
| Held-out hospitals | +0.013913 | 7/7 |
| Both checkpoints saw query | +0.008631 | 7/7 |
| Neither saw query | +0.012218 | 7/7 |

The observed own/full gap is not only a mismatch between MAE scoring and the
value loss used during fitting. These are descriptive differences, without
new significance claims. They exclude the latent JEPA objective, optimizer
trajectory and epoch-by-epoch fitting scores; no unique root cause or claim
that changing the loss cannot help follows. The selected checkpoints and query
exposure caveats from prior findings remain.

No runtime, calibration, fitting or promotion change. Evidence:
`artifacts/map_native_query_objective_20260912.json`.
