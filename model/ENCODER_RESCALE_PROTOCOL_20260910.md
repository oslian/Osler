# Source Numerical Coordinate Repair

Repeat the32-epoch four-arm/seven-seed trial, changing ONLY non-glucose numeric
standardization. For each source-fit partition, take unique snapshot indices
used in legal fit histories. Median and standard deviation on observed values
define a fixed per-channel affine transform in old normalized coordinates.
This follows the existing StateScaler standard method; no new library needed.
Keep glucose unchanged, missing normalized0, all masks/ages/time/presence intact.
Constant or unsupported channels retain identity. Source-seen restriction stays.
No external centering, clipping, guessed units, or external variable selection.

These snapshot statistics may weight carried observations more than raw events;
this is recorded, not a claim of patient-equal scaler estimation. Numeric
rescaling does not repair wrong units or analyte identity. Those contracts must
already be correct. Existing glucose teacher/output normalization is unchanged.

All other settings match encoder_budget. Own-only training logs and predictions
must reproduce because non-target slots are missing. Compare each full arm to
the corresponding original32-epoch predictions, plus whole-vs-own/persistence.
Fixed source-tune checkpoint selection, no MIMIC setting choice. All seven
exploratory paired patient95% CIs reported, not multiplicity-adjusted promotion.
Candidate checkpoints MUST carry centers/scales in addition to support mask,
activation and original scaler; no automatic runtime loading or old calibrator.
