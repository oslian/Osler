# Availability Encoder Bottleneck Diagnosis

Scope: glucose@6h direct neural decoder in the 2026-09-10 encoder pilot.
This does not diagnose every release source or prove a universal root cause.
No weights, serving, registry, intervals or promotion status changed.

## Confirmed Structural Limitation

`whole_body_joint_jepa.py::_decode_latent` implements normalized current plus
`0.75 * movement_gate * tanh(value_output)`. Since the gate cannot exceed one,
the release glucose scale40.030201 imposes an absolute30.022651 mg/dL delta cap.
This remains true with unlimited head activations and regardless of latent quality.
The characterization test forces both signs with extreme activations.

| Role | Pairs | True delta outside cap | Best possible bounded MAE |
|---|---:|---:|---:|
| Source development | 7140 | 45.94% | 23.025 |
| eICU hospital | 6077 | 46.29% | 25.262 |
| MIMIC validation | 6616 | 35.35% | 18.600 |

Outside-cap fractions are row-weighted. MAE is patient-equalized, mg/dL.
The bound uses future truth ONLY as an unattainable oracle, never as an input or
deployable policy. It is a lower error bound, not expected benefit of removing
the cap. Both own and whole arms have this cap, so it cannot alone explain their
performance difference. Actual whole MIMIC MAE36.091 remains far above18.600.

## Whole Context Transfer Is Not Just Insufficient Movement

Median over three trained seeds: on MIMIC, whole predicts absolute delta21.03
versus own13.23 mg/dL; whole has28.23% of rows above95% of the cap versus
own10.60%. Delta correlation is0.279 versus0.371. These are descriptive point
estimates, not new significant comparisons. Whole moves MORE, but tracks actual
changes less well. Blind amplification is therefore not a justified repair.

Same whole-trained weights with other variables made missing at inference:

| Seed | MIMIC full input MAE | Same weights, own-only input MAE |
|---|---:|---:|
| 7 | 36.691 | 34.390 |
| 19 | 36.091 | 38.136 |
| 31 | 35.622 | 35.888 |

All three full-input eICU hospital point estimates beat their masked-input
counterparts, whereas MIMIC masking helps only seed7. Input masking is itself a
distribution change. This localizes input/learned-response sensitivity, NOT a
unique harmful organ or proof that extra physiology is useless. Do not select
seed7 after inspecting MIMIC. Earlier support/scaling drift remains a possible
contributor, not an isolated causal explanation.

## Optimization Evidence

At selected seed7 checkpoints, three fixed source batches of64 rows per view:
whole context JEPA gradient norm is5.3%-7.8% of the value-loss gradient norm;
own7.6%-12.0%. Gradient cosines are positive (whole0.102-0.149), so sampled
gradients do not show direct conflict. The JEPA objective is active but much
weaker in these samples. Weight1 is not equal influence. This is not a history
of all training gradients or proof that increasing its weight improves outcomes.

## Diagnosis And Next Discriminating Repair

There are two separable failures: representation-to-change has a provable hard
amplitude restriction; use of extra context has seed/domain-dependent errors.
The earliest proven restrictive operation is the direct decoder, before scoring.
The earliest location of the context generalization defect remains unresolved
between encoding and future prediction; these data do not justify naming one.

Next controlled source-only trial: retain architecture, inputs, patients and
teacher; compare bounded versus an unrestricted signed-residual head with robust
loss and unchanged deployment gates. Separately measure/adjust JEPA gradient
balance, not both changes in one arm. Removing the cap risks extreme outputs;
reweighting JEPA risks hurting value accuracy. Fit/calibration must be redone.
Neither repair is completed or promoted by this diagnostic. No pristine external
lockbox was used; MIMIC remains repeatedly inspected validation data.

Artifacts: `artifacts/encoder_bottleneck_20260910.json`;
reproducer: `python -m scripts.diagnose_encoder_bottleneck`.
Uses existing local candidate checkpoints; contains no row-level output.
