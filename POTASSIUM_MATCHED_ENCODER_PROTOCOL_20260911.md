# Potassium Matched Encoder Diagnostic

Prior auxiliary checkpoints were selected for glucose. Test target-specific
supervision and equal-capacity own versus whole history on the existing measured
potassium pairs. This is an ablation of the existing model, not a parallel
runtime architecture. All original source patient partitions remain fixed.
No MIMIC, hospital or NWICU test reads. No target or threshold selection there.

Seven seeds, 64 epochs,16 patient-uniform batches of32 per epoch, AdamW1e-4,
weight decay1e-4, clip1. Same warm start, identity value activation, observed-step
belief mask, value SmoothL1 plus target-specific EMA JEPA loss. Whole support
is computed only on fit patients with potassium pairs. Own removes all other
input channels with the same missingness contract. No auxiliary losses.
Select each arm by potassium source-tune patient-equal MAE including epoch0.
Compare own/whole/persistence with patient-paired bootstrap1000, report all
seeds and source support. Checkpoints remain private. Selection-data CIs cannot
promote. Pretraining patient overlap remains unexcluded. These are still
glucose-anchored queries, not a general potassium cohort. No convergence claim
from an epoch limit and no inference that a failed comparison proves no signal.
