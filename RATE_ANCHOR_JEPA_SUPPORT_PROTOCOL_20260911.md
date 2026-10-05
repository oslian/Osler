# Train-matched numerical support control

Question: at the fixed rate-record potassium queries, does whole JEPA need
other lab numbers beyond their observation support? Earlier glucose-query
support controls do not test this newly trained task. Frozen neutralization
was off-manifold; this control trains with neutralization from epoch zero.

Reuse the exact original seven source patient partitions, initialization,
32 epochs, 16 steps, batch32, optimizer, patient sampler, EMA delta teacher,
identity residual, source support, and independent tune epoch selection.
Replace all supported non-potassium numerical history values by checkpoint
normalized zero; retain masks and ages. Potassium history stays unchanged.
Use hash-bound original own/whole predictions as controls; no new fitting on
evaluation or hospital patients. Report all-window and bracket-after scores.

Primary contrast whole versus trained support-only; secondary support versus
own and persistence, using patient-equal MAE and paired patient bootstrap1000.
Do not use hospital results for feature selection or routing. Overlapping seeds,
reused hospitals, pretraining overlap and retrospective rate-time limitations
remain. A negative control cannot establish information absence or causality.
No runtime, calibration or promotion change. This is a diagnostic of the same
base JointWholeBodyJEPAWorldModel, not full persistent organ-state validation.
