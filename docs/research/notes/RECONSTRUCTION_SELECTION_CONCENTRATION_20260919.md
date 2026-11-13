# Source Selection Score Concentration

Read-only replay of six completed real-model source trajectories: small and
expanded sequential, accumulated same/sqrt learning rate, decay-matched and
moment-clock. All recorded epoch macro scores and selected checkpoints reproduce
within1e-10. No evaluation labels, patient exclusion, weight updates or revised
deployment selector. Five focused tests pass.

## Findings

AST, CPK, ALT, bilirubin and lipase together account for50.20%-51.88% of the
selected source normalized MAE despite being5/33targets. Equal target counts
are not equal relative-error scales. This is score concentration, not proof
that these targets dominate shared gradients: training uses SmoothL1 rather
than the selection MAE and its derivatives differ.

| Model | Selected epoch | Top-five score share | Leave-one-target-out changes | Relative-error diagnostic epoch |
|---|---:|---:|---:|---:|
| Small sequential | 8 | 50.20% | 7/33 | 3 |
| Expanded sequential | 7 | 51.19% | 0/33 | 7 |
| Accumulated same LR | 8 | 51.88% | 0/33 | 8 |
| Accumulated sqrt LR | 7 | 50.20% | 0/33 | 5 |
| Decay matched | 7 | 50.59% | 0/33 | 7 |
| Moment clock | 5 | 51.12% | 0/33 | 5 |

Relative-error diagnostic divides each group-target error by its own epoch0
population-reference error, then averages. It is descriptive only, not a newly
validated metric or permission to deploy a retrospectively selected epoch.
Leave-one-out removes a whole target across both source groups, only to measure
sensitivity; no target was removed from the task.

In expanded sequential training, ALT+AST contribute .065399 of the total
.142824 initial-to-selected macro reduction (~45.8%). Twelve of66group-target
entries nevertheless worsen against epoch0. These contributions sum exactly
to the recorded macro change. Score concentration and target tradeoffs coexist.

## Interpretation

The concentration is real, but no single omitted target changes selection for
five of six trajectories. In particular, both expanded sequential and the
latest moment-clock model retain their selected epoch under reference-relative
scoring too. Thus this diagnostic does not support score rescaling alone as
the missing repair. It also does not invalidate improvements in these five
targets. Clinical importance cannot be inferred from error magnitude.

A loss-balancing intervention would change the training trajectory and remains
distinct from this source-selection replay; it requires measured training
gradient/scale effects and matched controls. Do not infer training domination
from large MAE alone or construct per-target checkpoint oracles. Previous
feasibility audit already shows no alternative shared checkpoint preserves all
observed gains.

Evidence: artifacts/native_reconstruction_selection_concentration_20260919.json.
All33trained/sixpending statuses remain. No runtime, interval or promotion change.
