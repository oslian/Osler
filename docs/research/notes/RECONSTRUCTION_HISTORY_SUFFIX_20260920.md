# Shared Reconstruction History Suffix

Fixed bounded-site epoch7; original source-selection patients (90 selected,
90 broad, with target-specific acceptance), 2470/2025 patient-target pairs,
all33 trained targets. This is not the expanded3323-person source replay.
Target and siblings are masked across the entire history before truncation.
All original full-history scores reproduce within1e-7.

| Group | Full history | Last8 events | Last1 event |
|---|---:|---:|---:|
| Selected |1.797984|1.808637|1.844191|
| Broad |1.787841|1.785876|1.812108|

Target-equal normalized current-reconstruction MAE, not physical-unit future
forecast errors. Last1 has numerical harm23/33selected and25/33broad;
last8 harms19/33in each group. History is not wholly unused. Last8 is mixed,
not an empirically justified universal history cutoff.

Important unresolved confound: truncation removes both repeat history AND
possibly the only observation of another variable. It also resets initial
state at a later event. Descriptive4000-draw paired intervals are unadjusted
and post-selection, not promotion gates or causal forgetting evidence.

Follow-up: preserve the latest observation event of every independent variable,
with/without the last8 events, then compare on identical source pairs. Do not
repeat the20260917 raw-summary/body ridge test as if it were untested.

Four suffix/bootstrap tests pass. No weights, runtime, intervals or promotion
changed. Report: artifacts/native_reconstruction_history_suffix_20260920.json.
