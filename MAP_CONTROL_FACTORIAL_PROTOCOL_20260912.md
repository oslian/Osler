# MAP Support Path Matched Training

Complete the two missing cells of the own/support-control history-by-query
factorial. Both mixed arms train from original initialization, not from the
control checkpoint used in the preceding off-distribution replay. All seven
seeds, patients, native histories, source-only transforms, 32 epochs, patient
sampler, AdamW, deterministic dropout and combined value/JEPA loss match the
original protocol. Source-tune selects epoch0..32 without evaluation tuning.

The existing own/own and control/control arms are frozen references. Train
control-history/own-query and own-history/control-query. Control means no
other numerical values, retaining support/age/process metadata and its native
activation cadence. This does not test full anatomical physiology or isolate
history masks from cadence. No new model selection based on hospital scores.

Compare each mixed arm against own and control using patient-equal MAE and
existing paired bootstrap CI. Record all contrasts, no new promotion rule.
All reused evaluation and hospital data remain diagnostic, not a new lockbox.
No runtime, intervals or release weights change. Private weights/predictions
stay outside version control; aggregate reports bind their source artifacts.
