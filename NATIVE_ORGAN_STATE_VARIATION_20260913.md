# Native Organ State Variation Localization

Source-only descriptive probe, same64fit/32tune patients and saved epoch8
none/sparse weights; initial seed7 no-edge weights provide a reference.
No training, fitting, external model selection or runtime changes.

On tune patients, centered RMS across patient owner-organ states is0.26142
(no-edge) and0.26325(sparse). Future organ states retain similar RMS0.26129 and
0.26307. The delta actually read by the head has RMS0.003873/0.004925, about
1.48%/1.87% of current-state RMS. Initial no-edge ratio was4.03%; training
reduces delta variation while mean delta norm increases0.34252->0.90730.
Fit patients show the same pattern: ratio3.88%->1.52% no-edge.

This localizes low patient-dependent variation to the generated future change,
not literal disappearance of all current-state variation. The future state
largely carries current differences while adding similar changes. These are
latent-coordinate statistics, NOT physiological units, information loss
percentages, accuracy evidence, or proof that current-state variation is useful.
Small variation alone can be rescaled; only predictive tests establish value.
Effective ranks/current/body variation are included in the machine report.

Together with head shuffling and matched anchor-only training, evidence now
points to a weak patient-dependent change readout in this particular pilot.
Next controlled repair must test conditional readout of current organ state
AND its change against anchor-only and original delta-only, maintaining a
single persistent trajectory, zero-start residual and source-only selection.
No unvalidated cross-organ edges should be enabled in runtime.

Evidence: `artifacts/native_organ_state_variation_20260913.json`.
