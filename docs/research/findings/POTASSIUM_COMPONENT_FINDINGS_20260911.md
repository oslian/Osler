# Potassium source component results

Seven frozen three-way whole checkpoints,262 evaluation patients per seed.
Full predictions reproduced at1e-6. Target current/history/mask/age unchanged.

| Intervention | Median MAE | Median prediction absolute shift | Improves full | Harms full |
|---|---:|---:|---:|---:|
| Non-target numbers neutralized | .379061 | .113457 |0/7|1/7|
| Non-target observed ages set3h | .374282 | .000633 |0/7|1/7|
| Non-target block missing | .406226 | .180790 |0/7|5/7|

Unchanged full medianMAE .374244; separate own model .377129. No intervention
beats separately trained own significantly in any seed. All comparisons are
descriptive/unadjusted, not a promotion gate; one age p-value does not establish
practical importance given tiny prediction shifts.

The cross-input path is used: block removal harms most fits. That does NOT
prove the numerical values supply stable transferable information. Neutralizing
numbers while preserving observation support yields a different response from
making the whole block missing. Exact age is a small local sensitivity. Effects
may include off-manifold activation or learned baseline changes; they cannot
uniquely attribute causality to hospital workflow or any organ. A constant-value
intervention is not a legitimate deployment input contract.

This is potassium-specific evidence beyond the earlier glucose component study.
It narrows the gap to how numerical information and support jointly contribute,
not a dead encoder path and not a demonstrated simple age or deletion repair.
To isolate useful numeric content, a future comparison must TRAIN matched
full-numeric and support-only controls on fit patients, select on tune and only
then score evaluation; fixed-weight off-manifold probes cannot establish that.
Keep the actual measurements in serving. No runtime, calibration or promotion.

Artifact: `artifacts/potassium_components_20260911.json` (complete=true).
Protocol: `POTASSIUM_COMPONENT_PROTOCOL_20260911.md`.
Command: `python -m scripts.audit_potassium_components`.
