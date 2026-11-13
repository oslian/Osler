# Native Replay Objective Control

Same original seed7 64fit32tune64evaluation64hospital pilot, eight epochs,
identity graph, zero-start heads, AdamW1e-4 and deterministic sampling.
Teacher frozen baseline for all arms (no EMA update); three comparisons:
standard normalized JEPA loss, centered-angular JEPA loss, value-only.
Centered loss subtracts FIT-only mean normalized teacher vector from normalized
predictions and targets, then uses existing normalized latent loss. This changes
geometry, not merely a mean-subtracted squared error which would cancel out.
Potential side effect: amplifying weak/noisy directions near the teacher center.
No forecast claims from latent loss alone. Tune selects epoch0..8; previously
inspected heldout patients/hospitals diagnostic only. No promotion/calibration.
Center and frozen-teacher policy stored with candidate checkpoints.
