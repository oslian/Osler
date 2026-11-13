# Potassium L1 Objective Trial

All14 paired64epoch own/whole fits completed. Only numerical SmoothL1 changed
to L1; JEPA, initialization, sampling, labels, patients, optimization budget
and selection remain fixed. No external data evaluation or runtime change.

Compared to each arm's original SmoothL1 checkpoint on identical source-tune
rows: significant improvement0/7 own,0/7whole; significant harm0/7both.
Median selected MAE own .36428 (original .36609), whole .34966 (original .34862).
Whole remains numerically better than own7/7 but significantly0/7. Both arms
beat persistence6/7; seed71 again fails. No change in promotion status.

This controlled intervention does not support objective alignment alone as
an effective repair. It does not prove all loss designs equivalent or that
SmoothL1 is theoretically optimal. Small noisy tune support and checkpoint
selection remain limitations. No new external sweep justified by this result;
do not relabel the change a successful fix or deploy it.

Artifact: `artifacts/potassium_mae_encoder_20260911.json`, complete=true.
Protocol: `POTASSIUM_MAE_PROTOCOL_20260911.md`.
Command: `python -m scripts.train_potassium_matched_encoder --mae`.
Private weights preserved outside Git. No changes to runtime/calibrators.
