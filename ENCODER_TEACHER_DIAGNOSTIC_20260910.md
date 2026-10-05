# Teacher Delta Discriminability

Read-only diagnostic of the 14 original 32-epoch own/whole unbounded
checkpoints in the encoder budget experiment. No fitting, source selection,
patient data output, release modification, or promotion.

Used the actual `teacher_target` function on glucose changes of
-120, -60, -30, 0, 30, 60, 120 mg/dL, scaled by the release glucose scale.
All other target values were masked. Both arms used seven seeds.

Results across the 14 checkpoints:

- Cosine similarity, -120 versus +120: -0.063 to 0.044.
- Cosine similarity, 0 versus +30: 0.935 to 0.955.
- Cosine similarity, +30 versus +120: 0.726 to 0.774.
- Centered seven-point teacher matrix numerical rank: 6 in every checkpoint.
- Constant zero-change teacher prediction normalized SmoothL1 loss:
  0.002308 to 0.002828 over this artificial grid.

Normalization removes embedding norm, but does not remove all physical
amplitude information: direction in embedding space changes with glucose
delta. This rules out literal constant teacher encoding on this grid, not
all possible collapse or insufficient supervision on real patient data.
The small loss is not independently evidence of a bug: unit normalization
and averaging across latent dimensions affect its numerical scale. Earlier
source gradient-balancing experiments already tested a related intervention
without resolving whole-versus-own transfer.

The current strongest remaining localization is the response to non-target
numeric context. Earlier fixed-weight numeric neutralization helped MIMIC
7/7, not eICU hospital; source affine rescaling did not repair transfer.
This is not proof of a single biological or coding root cause. Measurement
method, cohort differences, and learned interaction generalization remain
unseparated. Results apply only to the tested glucose@6h candidate.
