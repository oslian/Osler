# Frozen Body Linear Probe

Completed both selected sampling checkpoints x35 target-specific real/placebo
ridge probes. Original target heldout MAEs reproduced to1e-7. Scaler and ridge
fit only on source-fit patients, alpha/population selection source-tune only.
No encoder weights updated. Patient histories remain target/sibling masked.

No real probe has supported95% paired improvement against population, original
readout or patient-deranged placebo in either heldout role, for either encoder.
Unsupported low-patient CIs do not imply equivalence or a successful gate.

Thus replacing the small near-constant neural readout by a source-selected
linear probe did not recover stable useful information in this pilot. The
claim that ONLY the existing readout prevents success is not established.
This does not prove latent information absent or all readouts impossible:
sample support is small, ridge is limited, source-selected encoders and tune
patients have already been used. No new untouched external evidence.

Next model changes must account for the limited learned-state evidence rather
than simply attributing failure to zero-start heads. Full shared-state training
and larger patient-level supervision remain unfinished. No promotion, runtime,
interval or saved deployment weights changed.

Evidence: artifacts/native_reconstruction_linear_probe_20260913.json.
