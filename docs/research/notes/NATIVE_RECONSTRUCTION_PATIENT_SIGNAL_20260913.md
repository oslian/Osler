# Trained Reconstruction Uses Mostly Small Common Corrections

Loaded both selected sampling-pilot checkpoints and private examples; all
parent hashes verified. Every heldout target MAE reproduces to1e-7. No updates.
Permuted same-target patients within each role by a fixed cyclic derangement
(one reservoir example per patient/target, so no self donor). Also report a
constant equal to mean fit predictions, never fitted on heldout targets.

Across-target medians, in inherited normalized target units:

| Arm/role | Mean absolute correction | Prediction std across patients | Permutation shift |
|---|---:|---:|---:|
| Event/evaluation |.03004|.00451|.00392|
| Event/hospital |.02866|.00548|.00517|
| Balanced/evaluation |.01064|.00137|.00154|
| Balanced/hospital |.01080|.00174|.00165|

The selected numerical readouts vary little across patients relative to their
already small offsets from fit medians. Thus the current pilot's tiny tune
gains do not demonstrate strong patient-specific reconstruction. This is output
near-constancy, not proof that body latents collapsed or contain no information.
Selected epochs differ(4vs2), so arm differences cannot be solely assigned to
sampling. Initial zero readout, short training, support and encoder usefulness
remain possible contributors. Do not declare a universal JEPA root cause.

Fit-mean constant is a frozen-model diagnostic, not an independently trained
capacity-matched baseline. No new hypothesis test or promotion claim. All
figures are normalized, not directly mmHg/mg/dL or clinical error estimates.

Evidence: artifacts/native_reconstruction_patient_signal_20260913.json.
No weights, calibration, serving or runtime changed.
