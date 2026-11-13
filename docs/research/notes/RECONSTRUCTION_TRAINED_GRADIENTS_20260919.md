# Trained Shared-State Gradient Probe

Expanded selected real checkpoint7, source-fit patients only. All33targets x
2sourcegroups x2disjoint eight-patient samples:1056 example gradients. Parameters
are8320 scalars in trajectory.layer.body_update_gate only, not the full encoder,
readout, organ transitions or full optimizer state. Gradients average SmoothL1
per patient in float64, matching the training loss. No optimizer step occurred.

Four targets have negative same-target cross-group cosine in both probes:
chloride,lactate,magnesium,potassium. The equal-task mean gradient has a negative
dot product with six task gradients in both probes: selectedBUN/ potassium;
broadCPK/lipase/platelets/totalprotein. This is a local first-order SGD conflict
indicator, not measured Adam harm or clinical physiological opposition.

Important counterevidence to a universal-conflict explanation:
- glucose cross-group cosines+.441,+.006: not opposed in either probe;
- troponin+.872,+.761: strongly aligned despite retention problems;
- AST+.298,-.120 andBUN+.141,-.031: signs do not replicate.
- Median same-task between-probe cosine is only.098;26/66 are negative.
  Broadglucose between-probe cosine-.270, selectedglucose+.352.

Small patient subsets yield unstable estimates even for the same task. This
does not establish irreducible noise, absent physiology or the cause of training
failure. It means a gradient-surgery intervention justified only by these
small-sample cosines would be premature. First estimate reliability as patient
sample size grows, still using source data, and compare actual measured updates
on disjoint probe patients before claiming persistent conflict. Current training
uses per-example sequential Adam, not this equal-task simultaneous SGD update.

Two tests verify cosine/dot signs and zero-gradient behavior. No raw identifiers
in artifact, no model/registry/interval/runtime change, no promotion.
Evidence: artifacts/native_reconstruction_trained_state_gradients_20260919.json.
