# Frozen Potassium Transfer Diagnostic

Evaluate all14 fixed target-matched checkpoints without selection or refitting.
Use the existing eICU hospital and MIMIC availability-safe glucose query slices
(fresh glucose with measured future, as in source construction), then rebuild
potassium current/future pairs: age<=6h, future5-7h, query horizon6h. Keep
all potassium-paired rows. Require no patient overlap with source development.
Retain source checkpoint support masks and scaler; no external normalization.

Compare whole versus own and each versus persistence, patient-equal MAE and
paired bootstrap1000. Report all seven seeds; care-unit groups with>=30patients
also receive descriptive CIs. No external recalibration, gate tuning or weights
changed. MIMIC was inspected repeatedly; this is held out of this training, not
an untouched lockbox. Pretraining overlap is not excluded. No promotion from
this uncalibrated diagnostic. Fix this protocol before reading predictions.
