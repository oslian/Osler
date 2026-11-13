# MAP Native Query Gradient Geometry

Inspected seven selected own/full checkpoints, three identical source-fit
patient-sampled probe batches per split/arm: 42 probes. Reconstructed input
transforms and checkpoint predictions reproduce saved outputs within float
tolerance before differentiation. No optimizer step or weight change.

Parameter scope: belief_filter, variable_mixer, module_mixer, state_encoder.
Value loss is SmoothL1; latent loss is the target-specific EMA JEPA loss with
its actual unit coefficient. Gradients are compared before Adam preconditioning.

| Arm | Opposing gradients | Median JEPA/value norm | Minimum raw value-descent factor |
|---|---:|---:|---:|
| Own | 19/21 | 0.014444 | 0.992696 |
| Full | 9/21 | 0.015168 | 0.994908 |

The factor is 1 + dot(value_gradient, latent_gradient)/norm(value_gradient)^2.
It stays positive and near one at these probes: no dominant opposing latent
gradient at the inspected shared blocks. This does not establish that latent
training is harmless globally, nor characterize early training, other parameters
or Adam's update direction. A loss ablation would be a different experiment.
The current evidence does not justify asserting "JEPA loss overwhelms value
learning" as the cause or deleting JEPA training as an established repair.

No training, runtime, calibration or promotion change. Evidence:
`artifacts/map_native_query_gradients_20260912.json`.
