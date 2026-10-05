# Native Organ Replay Paired Pilot

Question: does training the new native observation/assimilation path produce
an incremental MAP forecast signal, and does opening the predefined anatomical
graph help beyond the same no-edge state model? This differs from old
initialize/rollout-only organ exchange; neither outcome guarantees deployment.

Fixed seed7, original MAP native task and partitions. Sample one eligible query
per patient with a fixed RNG, then64fit/32tune/64new-source/64hospital patients.
No future magnitude, model error or event-count selection. This is explicitly
a small integration/learning pilot, not the full cohort or seven-seed gate.

Arms: identity adjacency versus all existing canonical anatomical pairs plus
self-loops, row-normalized. Same initialization and patient order,8epochs,
AdamW1e-4, weight decay1e-2 (PyTorch default), clip1, value SmoothL1 plus EMA
target-specific future-latent loss. Exact zero-residual epoch0 is eligible;
select by source-tune patient MAE only. No body-feedback bypass. Non-owner
corrections are scoped in both arms. Actual body utility is not separately
supervised in this pilot and must not be claimed.

Use source-trained own-history JEPA as frozen baseline per query. Candidate
uses a trainable copy of the same JEPA, recovered native clocks, all source-
supported measurement values and local pre-mixer organ proposals. Treatments
are excluded, not claimed complete. Fixed nominal6h query retains the original
label task. No post-anchor event, hospital-derived transform or outcome routing.

Report all selected and epoch8 source-tune performance, plus patient-paired
20,000bootstrap comparisons versus baseline and persistence in both held-out
roles, and sparse versus identity. Reused evaluation is not a new lockbox.
No multiple-comparison claim of promotion, intervals or clinical utility.
Do not change the protocol after results. Checkpoints/patient rows stay private.
