# Actual Multi-Target Sampling Training

Completed both8epoch/512update arms on the same raw-event current reconstruction
contract.32fit/16tune/16evaluation/16hospital patients;35eligible targets,
879/442/422/427 reservoir patient-target examples. All targets share body/model
parameters. Each sample masks its full target/sibling history. Initial readout
is fit target median with zero weights, not a random numerical predictor.

Event-frequency weighted reservoir: tune normalized target-equal MAE
1.811102 ->1.808488, selectedepoch4. Balanced target/patient draws:
1.811102 ->1.810791, selectedepoch2. Both improve tune only slightly.

No target has a supported95% paired bootstrap improvement versus population
in either heldout role for either arm. Balanced versus event-weighted also
has no supported improvement. Some small-support comparisons return no CI;
these are unsupported, not demonstrated equivalence. No significant harm
was found by the same checks, but this small pilot cannot establish safety.

This is actual multi-target training, beyond the previous gradient smoke.
However, sampling balance alone does not establish masked shared-state recovery.
It does not prove shared information absent or negate future forecast evidence.
Finite-budget/low-patient-support and current-only supervision remain limits.
Event-weighted reservoir is not literal replay of every observed event.
No matched placebo/full7seed/conformal/promotion test was completed here.

Private sample cache and selected weights were saved for reproducible followup;
paths/hashes are bound in artifacts/native_reconstruction_sampling_20260913.json.
No release weights, registry, intervals or serving behavior changed.
