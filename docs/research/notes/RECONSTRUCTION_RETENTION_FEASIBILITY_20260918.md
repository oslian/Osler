# Can Selection Alone Repair Retention?

Exact source-selection score trajectories, no new weights/labels. Across all66
group-target pairs, epoch0 is the only recorded checkpoint with no numerical
harm relative to the initial equal-group population prior. Every trained epoch
harms7-12targets per group. A strict all-target numerical selector would simply
retain the untrained predictor, not repair the shared state.

Selected-group targets that never beat initial prior in any of8epochs:
CPK,glucose,lipase,troponin. Broad-group:CPK,temperature,troponin.
Some are low support; per-target support is retained in the artifact.
Selected targets with an improving epoch but harmed at chosenepoch6:
calcium,fibrinogen,magnesium,MAP,RR. Broad:HR,magnesium,RR,triglycerides,WBC.

Therefore checkpoint selection is not sufficient even with optimistic per-target
choices. Some failures precede selection. This does NOT prove intrinsic target
unpredictability, absence of usable physiology, gradient conflict, or that longer
training cannot help. The source-selection labels have already selected epoch6;
per-target minima cannot become independent evidence or deployment heads.
No new acceptance criterion introduced: numerical zero harm is a diagnostic,
not a statistically justified noninferiority threshold. Three tests pass.

Next repair must address training/representation of never-improving targets,
while measuring retention separately for transient improvements. Preserve all
targets, both cohorts, existing masked-history rules and external gates.
No runtime/interval/promotion change.
Evidence:artifacts/native_reconstruction_joint_retention_feasibility_20260918.json.
