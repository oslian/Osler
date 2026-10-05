# Expanded Source Selection Replay

Freeze the expanded sequential real training trajectory (epoch0 plus1-8).
Do not refit or change gates. Evaluate every trained target on all3323existing
broad encoder_tune patients, retaining original90 and additional3233 as separate
summaries. Verify original90 scores against recorded training at every epoch.
Use existing roles; exclude all fit and original selected-group patients.
No MIMIC, NWICU or evaluation-role labels enter this replay.

Report each target's errors, macro errors and epoch rankings for original,
additional and combined broad source selection. Also report equal-group macro
combining full broad with the unchanged original selected-group selection scores.
Epoch0 uses exactly the recorded source-fit median initialization. Six untrained
targets remain explicitly pending, not scored as trained. Preserve checkpoints
and per-patient predictions privately for paired follow-up.

This diagnoses dependence on a small source selection cohort, not independent
external validation, a causal sample-size experiment or promotion. Additional
patients differ in composition as well as number. No source-selected replacement
may be deployed without its remaining full gates. Do not choose targets, thresholds
or alternative metrics after seeing these results. Original scores/weights remain.
