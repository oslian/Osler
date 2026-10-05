# Selection Scores Versus Local Gradients

Read-only interpretation of the saved expanded epoch7 body-update-gate probes.
Verified both source-report bindings and reconstructed every recorded gradient
dot product from saved norms and cosine matrices to1e-12 absolute/1e-10 relative
tolerance. No patient data were reread, no new model was fit, and no evaluation
labels or thresholds were used. Six focused tests pass.

The five largest selected-score contributors (AST, CPK, ALT, bilirubin, lipase)
account for51.19% of source selection MAE. In the two saved disjoint eight-patient
per-task probes, their shares of summed task gradient norms are22.12% and20.65%.
Their signed projections onto the equal-task mean gradient contribute19.11%
and11.73% of the summed projections. Projection shares are signed algebraic
quantities, not probabilities or causal influence percentages.

These five targets represent10/66group-target tasks (15.15%). They are moderately
overrepresented in norm mass, but do not account for most local gradient mass.
Troponin is among the largest norm tasks in both probes despite not belonging
to the five highest score contributors. The norm rankings otherwise vary.

This weakens the proposed shortcut from 'large selection MAE' to 'these five
targets dominate all learning'. It does not establish balanced full-model
training: only8320body-update-gate parameters at one checkpoint are inspected,
not readout/organ/encoder blocks or sequential Adam steps. The previous larger
sample reliability audit showed small-probe signs are unstable. Do not derive
new target weights from these small-sample rankings or claim a unique root cause.

SmoothL1 training and MAE selection have different derivative behavior; measured
gradient evidence is required before treating error scale as update scale.
Selection score concentration remains a real descriptive property, but neither
this result nor the earlier leave-one-target-out replay demonstrates a repair.

Evidence: artifacts/native_reconstruction_score_gradient_concentration_20260919.json.
Original33trained/sixpending targets retained. No runtime, interval or promotion
changes. Broader source exposure and matched full-training controls remain
separate unfinished work, not disproved by this local probe.
