# MAP Value / JEPA Gradient Geometry

Selected own/full checkpoints, seven source splits, three patient-uniform
source-fit batches per split/arm. Same batches within each split. No optimizer
updates or tuning. Parameter scope: belief_filter, variable_mixer,
module_mixer, state_encoder. Parents and event transforms verified.

| Arm | Negative cosine probes | Median cosine | Median JEPA/value norm ratio |
|---|---:|---:|---:|
| Own | 11/21 | -0.00905 | 0.01569 |
| Full | 13/21 | -0.05556 | 0.01530 |

Maximum norm ratios are0.02376own/0.02646full. For the raw summed gradients,
the value-gradient inner product normalized by its own squared norm is
`1 + ratio * cosine`; minimum0.99342own/0.99172full. None reverses that local
value descent direction. This is raw-gradient geometry, NOT Adam's actual
preconditioned update or a proof of finite-step improvement.

There is local opposition but no evidence here that a dominant JEPA gradient
overwhelms the value objective. This does not prove removing/increasing JEPA
loss would have no effect. Initial/intermediate weights, other parameter groups
and accumulated training effects remain outside this diagnostic. Do not promote
a loss adjustment on this basis or claim the unique root cause is established.

No weight/runtime/calibration changes. Evidence:
`artifacts/map_loss_gradients_20260912.json`. All42 probes source-fit only.
