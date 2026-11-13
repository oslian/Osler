# Source-Selected Native MAP Refinement

All seven full models have better minimum tune MAE in epochs 17-32 than 1-16.
This does not prove undertraining but motivates a matched refinement test.
Freeze the following policy before running refinement:

Start each arm from its own original source-tune-selected checkpoint; fresh
AdamW state (original optimizer state was not saved), unchanged 1e-4 learning
rate and weight decay; 32 additional epochs of 16 x 32 patient-balanced draws.
Use seed+10000 for a fresh, identical patient/query sequence in own/full arms.
Retain identical inputs/transforms, JEPA/value objective, deterministic dropout
policy and gradient clipping. Select refinement epoch zero or later only on
the same complete source-tune patients. No held-out model selection.

Compare refined full versus refined own, original full, and persistence;
compare refined own versus original own. Same patient-level paired bootstrap
in source evaluation and hospital. Seven seeds, no auto-promotion. The reused
tune/hospital data are not fresh confirmation. This tests extra refinement
with optimizer reset, not uninterrupted longer training or isolated Adam-reset
effect. Original models remain untouched; artifacts/checkpoints use new paths.
