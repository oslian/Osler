# Fixed Context Dropout Trial

Question: does end-to-end training over-rely on non-target numerical/support
combinations that fail to transfer? This differs from post-training input
neutralization and affine rescaling. Use the existing masking/unknown contract
as training augmentation, no new architecture, dependency or inference rule.

Repeat original32-epoch seven-seed own/whole bounded/identity trial, with
probability0.5 variable dropout in whole-context training only. One independent
Bernoulli per sample and non-glucose variable; apply to all history times and
current observation together. Set value0, mask0, age7 and recompute presence.
Never mask glucose, alter truth, change query time or use external statistics.
No inverse-probability value scaling: these are missing observations, not
activation dropout. Sampling uses a separate fixed random stream so patient
batch sequence is identical to the original experiment.

Evaluate on full legal inputs without augmentation. Source-only support mask,
original numeric scale, source fit/tune split, optimizer,32epoch budget and
selection including epoch0 remain identical. Own controls must reproduce all
logs and predictions exactly.28 fits total. No dropout probability search.

Primary comparison: paired patient MAE against original same-arm predictions
and matched own arm, separately in eICU hospital and MIMIC, all seven seeds.
Also report persistence, latent rank and existing descriptive care-unit/time
strata. Improved tune loss alone is not success. Both external datasets have
already been inspected; this is validation, not untouched lockbox evidence.
No runtime weights, uncertainty calibration or promotion is authorized by this
diagnostic. A null result rejects this fixed augmentation, not every form of
regularization. A positive result does not prove biological causal coupling.
