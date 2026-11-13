# Expanded Checkpoint Feasibility

Source-only replay of all nine recorded real-model checkpoints (initial plus
eight trained epochs), both conditions and all66 group-target combinations.
No evaluation labels, new fitting or revised gate.

Small: only epoch0 is numerically nonharming every group-target relative to its
prior. Every trained epoch harms12-18 of66. No epoch other than selected8
weakly dominates that selected checkpoint on all66 scores.

Expanded: only epoch0 is numerically nonharming every group-target relative to
its prior. Every trained epoch harms12-17 of66. No epoch other than selected7
weakly dominates selected7. At selected7,10 targets/group entries had earlier
numerical gains but lost them: selected fibrinogen,glucose,lactate,magnesium,
temperature,troponin; broad CPK,glucose,triglycerides,troponin. Selected CPK and
broad temperature never beat their own prior during the eight trained epochs.

Thus choosing a different single existing checkpoint cannot numerically retain
every current gain while repairing others. This is not a claim that all trained
models are useless, nor a statistical noninferiority standard or proof that no
future training can succeed. Tiny source gains and sparse counts remain noisy.

Historical output-retention experiments were reviewed: uniform and source-
selected retention improved some electrolyte outcomes but harmed renal outcomes
(NATIVE_RETENTION_REPAIR_FINDINGS_20260911.md and
NATIVE_SELECTED_RETENTION_FINDINGS_20260911.md). Do not present frozen-output
retention as an entirely new guaranteed solution. Those were different native
forecast tasks, not this expanded masked-current-reconstruction task.

The next distinct intervention should target shared-update interference only if
measured on this trained state/source data, with a matched no-intervention arm.
Earlier initial random-state gradient cosines do not establish trained conflict.
Training loss balance, source diversity and retention constraints remain hypotheses.
Do not construct an oracle collection of per-target best epochs from these logs.

Evidence: artifacts/native_reconstruction_exposure_selection_feasibility_20260919.json.
Two trajectory tests cover stable ordering, macro-improves/target-harms behavior
and nonfinite rejection. Runtime and promotion unchanged; all39 statuses retained.
