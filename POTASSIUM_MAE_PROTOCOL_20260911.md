# Potassium Value Objective Alignment Trial

Single change from the completed matched trial: numerical SmoothL1 becomes L1
on the same normalized target. JEPA loss and its coefficient remain unchanged.
Use existing PyTorch loss implementations. Seven seeds x own/whole,64epochs,
same initialization, fit/tune patients, labels, sampler, optimizer, mask and
checkpoint selection. No hyperparameter sweep. No external data reads. Retain
all prior results, use fresh candidate output paths.

Compare each arm against its prior SmoothL1 predictions on identical tune rows,
and whole versus own and persistence with paired patient bootstrap1000. The
sampler already gives each fitting patient equal expected weight. L1 aligns
the point objective with MAE but may reduce smooth optimization or prioritize
central outcomes at the expense of rare changes. This is not a guaranteed bug
fix, and cannot replace separate risk/uncertainty evaluation.

No source-tune result alone promotes. No threshold relaxation, no architecture
or runtime change, no claim this one target settles whole-body learning.
