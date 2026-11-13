# Fixed shared future training

Same native glucose cohort, own/whole views, seven seeds,32epochs16x32 patient
uniform updates, AdamW1e-4, source-tune checkpoint selection including zero.
Only latent_predictor is frozen; all original trainable parameters remain
trainable, including history input projection. EMA encoder updates normally.
No retention loss, architecture change or hospital-based selection.
Compare matched own and original own/whole/persistence. Preserve all-target
predictions privately for actual-label retention evaluation, especially BUN.
Frozen future weights must remain exactly identical. Other outputs need not
remain equal because history/context still learn. No deployment/calibration
promotion, new lockbox or causal claims. Base-model diagnostic only.
