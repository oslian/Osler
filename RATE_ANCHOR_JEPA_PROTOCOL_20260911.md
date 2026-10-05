# Shared encoder at record-triggered queries

Seven fixed patient partitions, own vs whole,32epochs x16steps x32batch,
patient/query sampling; identical initialization, optimizer and teacher loss.
Train actual context encoder with observed-step masks, source-supported channels,
identity value residual and EMA target-specific potassium future-delta teacher.
Tune patients select epoch including zero; evaluation/hospital never select.
Report patient paired MAE versus own and persistence, all/bracket-after groups,
and descriptive latent rank. Not full promotion or untouched external testing.

No treatment fields added to lab slots. This tests whole-lab representation at
new queries before adding an event-state adapter; no claim of treatment learning.
Warm-start pretraining patient overlap not excluded. Fixed6h query with5-7h
measured labels; not exact-time forecasting. Other-target retention, intervals,
runtime gates unfinished. Persistent-organ architecture not tested by this base
JointWholeBodyJEPAWorldModel contrast. No release modification.
