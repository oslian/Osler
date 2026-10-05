# Observation-Aware Encoder Fine-Tuning Pilot

Fixed glucose@6h availability queries, same patients, actual5-7h measured labels
from the completed input audit. Reuse raw qualified lab event caches and exact
lag snapshot construction. Do not use future actual time as a predictor; nominal
6h request remains fixed. Unknown bedside/treatment values remain unknown.

Four arms: own/full labs x frozen/trainable context encoder. Same existing
checkpoint, input schema, neural future/value decoder, identical minibatches.
Not a new311-feature ridge. All use truthful mask/age/organ presence. Context
prefixes are those used by the existing reconstruction audit. End-to-end arms
unfreeze these; remaining online forecasting components train in all arms.
EMA teacher retains stop-gradient, updates after optimizer steps. Supervise the
glucose signed-delta future latent using the existing target encoder and normalized
JEPA loss plus normalized value SmoothL1. No other future targets fabricated.

Three-seed bounded pilot (7,19,31), not full seven-seed promotion. Eight epochs,
16 steps of32 patient-balanced samples per epoch. AdamW1e-4, weight decay1e-4,
gradient norm1. Deterministic eval-mode dropout in all training arms. No learning
rate or hyperparameter sweep. Source-tune patient-equal MAE selects checkpoint,
including epoch0. Retain best actual weights and context gradient/change audit.

Compare own/full, frozen/trainable, initial direct neural model, persistence.
The current shared external head is not refitted or attached to changed latent;
direct value forecasts are the matched neural endpoint, not the previous ridge
or311-feature shared-head predictions. Report latent effective rank/std.
Pretrained encoder patient overlap has not been ruled out; new fitting uses
eICU fit patients only. MIMIC already inspected; never claim untouched lockbox.

This tests warm-start encoder adaptation, not random-initialization learning,
complete continual self-learning, intervals, other-target retention or a full
promotion gate. No serving, registry, release checkpoint or calibrator changes.
