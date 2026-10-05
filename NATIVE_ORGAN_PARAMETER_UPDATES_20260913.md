# Native Replay Parameter Update Audit

Compared exact retained selectedepoch0 and finalepoch8 parameters for both
none/sparse arms, trace bindings checked. Relative parameter L2 movements:

| Group | None | Sparse |
|---|---:|---:|
| Local observation encoding |0.64%|1.23%|
| Organ observation update |8.56%|9.09%|
| Organ dynamics/messages |8.64%|8.31%|
| Body observation update |0%|0%|
| Body dynamics |0%|0%|
| Value head |4.33%|4.34%|
| Latent head |57.99%|58.06%|

These norms include weight decay and incomparable parameter scales; they are
not learning allocation percentages. Organ parameters do move, so literal
disconnected organ optimization is not supported. Body-specific parameters are
unchanged exactly, consistent with body_feedback=False and owner-only heads.
This was a deliberate no-bypass graph control, not newly discovered proof of
why all previous forecasts fail. It also means this pilot cannot validate
learning of the complete body state merely because the body is maintained.

To pursue the full objective, body state requires an explicit legal supervision
path, with masked reconstruction/trajectory tests and owner-only forecast guards
preserved. Do not simply enable unrestricted global mixing to get gradients.
Before implementation, reuse existing body reconstruction contracts and ensure
their targets cannot leak through the shared encoder input.
No new training, runtime, calibration or promotion changes.

Evidence: `artifacts/native_organ_parameter_updates_20260913.json`.
